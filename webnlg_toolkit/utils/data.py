import re

from sklearn.utils import resample

from webnlg_toolkit.utils.benchmark_reader import Benchmark, Tripleset, Triple, select_files


def load_webnlg_xml(loc, return_type='list'):
    b = Benchmark()
    xml_file = select_files(loc)
    b.fill_benchmark(xml_file)

    corpus = b.to_dict()["entries"] if return_type == 'list' else b

    return corpus

def get_lang_count(loc):
    corpus = load_webnlg_xml(loc) if isinstance(loc, str) else loc

    langs = {}
    for entry in corpus:
        for lex in entry["lexicalisations"]:
            if lex["lang"] not in langs:
                langs[lex["lang"]] = 1
            else:
                langs[lex["lang"]] += 1
    return langs

def from_camel(s):
    return re.sub('([A-Z])', r' \1', s).lower()

def to_camel(s):
    s = re.sub(r"(_|-)+", " ", s).title().replace(" ", "")
    
    # Join the string, ensuring the first letter is lowercase
    return ''.join([s[0].lower(), s[1:]])

def parse_tripleseq(s):
    s = re.split("(<S> |<P> |<O> )", s)[1:]

    triples = []
    buff = {"subject": "", "property": "", "object": ""}
    for i in range(0, len(s)):
        if s[i] == "<S> ":
            buff = {"subject": "", "property": "", "object": ""}
            buff["subject"] = s[i+1]
        elif s[i] == "<P> ":
            buff["property"] = to_camel(s[i+1])
        elif s[i] == "<O> ":
            buff["object"] = s[i+1]
            triples.append(buff)

    return triples

def dict_to_tripleset(ds):
    ts = Tripleset()
    ts.triples = [Triple(d["subject"], d["property"], d["object"]) for d in ds]

    return ts

def inject_generated_triplesets(b, outputs):
    for i, entry in enumerate(b.entries):
        t_dict = parse_tripleseq(outputs[i])
        t_set = dict_to_tripleset(t_dict)
        entry.generatedtripleset = t_set

    return b

def strip_entity(s):
    return s.replace("_", " ").strip('\"')

def is_lang(subject, lang):
    if lang == "all":
        return True
    return subject == lang

def linearize_triple(triple):
    s = strip_entity(triple['subject'])
    p = from_camel(triple['property'])
    o = strip_entity(triple['object'])

    return f"<S> {s} <P> {p} <O> {o}"

def linearize_triples(triples):
    return "\n".join([linearize_triple(triple) for triple in triples])

def linearize_corpus(corpus):
    return [linearize_triples(x["modifiedtripleset"]) for x in corpus]

def get_input_ref_pairs(corpus, lang="all"):
    inputs = linearize_corpus(corpus)
    references = [[(l["lex"], l["lang"]) for l in x["lexicalisations"] if is_lang(l["lang"], lang)] for x in corpus]

    return list(zip(inputs, references))

def load_webnlg_dataset(loc, lang="all", multilingual=False, task="rdf2text", training=True):
    corpus = load_webnlg_xml(loc)
    pairs = get_input_ref_pairs(corpus, lang=lang)

    # text2rdf is treating like training because it never has multiple references
    if training or task == "text2rdf":
        # flatten refs
        flat_pairs = []
        for pair in pairs:
            for ref in pair[1]:
                flat_pairs.append((pair[0], ref))
        pairs = flat_pairs

        if multilingual:
            # upsample minority languages
            lang_counts = get_lang_count(corpus)
            max_count = max(lang_counts.values())
            for lang, count in lang_counts.items():
                if count < max_count:
                    samples = [pair for pair in pairs if pair[1][1] == lang]
                    pairs += resample(samples, replace=True, n_samples=max_count-count, random_state=123)

            if task == "rdf2text":
                pairs = [(f"RDF-to-text in {pair[1][1]}: " + pair[0], pair[1][0]) for pair in pairs]
            elif task == "text2rdf":
                pairs = [(f"Text-to-RDF in {pair[1][1]}: " + pair[1][0], pair[0]) for pair in pairs]
        else:
            pairs = [(pair[0], pair[1][0]) if task == "rdf2text" else (pair[1][0], pair[0]) for pair in pairs]
    else:
    
        if multilingual:
            # assumes that task == "rdf2text"
            pairs = [(f"RDF-to-text in {lang}: " + pair[0], pair[1]) for pair in pairs]
        else:
            pairs = [(pair[0], pair[1]) if task == "rdf2text" else (pair[1], pair[0]) for pair in pairs]

    return pairs