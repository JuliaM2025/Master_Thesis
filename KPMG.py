import os
import pandas as pd
import re
import nltk
from nltk import pos_tag, word_tokenize
from collections import Counter
from nltk.corpus import stopwords
from datetime import datetime
import spacy

# Load a German NLP model for Named Entity Recognition
nlp = spacy.load('de_core_news_sm')  # Ensure this model is installed
print("Current working directory:", os.getcwd())

# Ensure nltk resources are available
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')

# Path to the folder containing .txt files
folder_path = r'C:\Users\danie\OneDrive\Desktop\julia\KPMG'


german_stopwords = set(stopwords.words('german'))

def clean_text(text):
    """Remove specific statements added at the end of each text."""
    unwanted_statements = [
        "OTS-ORIGINALTEXT PRESSEAUSSENDUNG UNTER AUSSCHLIESSLICHER INHALTLICHER VERANTWORTUNG DES AUSSENDERS - WWW.OTS.AT | KPM",
        "© 2022 KPMG Austria GmbH Wirtschaftsprüfungs- und Steuerberatungsgesellschaft, eine österreichische Gesellschaft mit beschränkter Haftung und ein Mitglied der globalen KPMG Organisation unabhängiger Mitgliedsfirmen, die KPMG International Limited, einer private English company limited by guarantee, angeschlossen sind. Alle Rechte vorbehalten.",
        "Rückfragen & Kontakt\nKPMG Austria GmbH\nLisa Kannonier\nPorzellangasse 51\n1090 Wien\n+43 664 8213 664\nlkannonier@kpmg.at",
        "Rückfragehinweis:\nKPMG Austria GmbH\nLisa Kannonier\nPorzellangasse 51\n1090 Wien\n+43 664 8213 664\nlkannonier@kpmg.at",
        "Über KPMG \nAls Verbund rechtlich selbstständiger, nationaler Mitgliedsfirmen ist KPMG International mit ca 236.000 Mitarbeiter:innen in 144 Ländern eines der größten Wirtschaftsprüfungs- und Beratungsunternehmen weltweit. Die Initialen von KPMG stehen für die Gründerväter der Gesellschaft: Klynveld, Peat, Marwick und Goerdeler.\nIn Österreich ist KPMG eine der führenden Gruppen in diesem Geschäftsfeld und mit mehr als 1.800 Mitarbeiter:innen an acht Standorten aktiv. Die Leistungen sind in die Geschäftsbereiche Prüfung (Audit) und Beratung (Tax, Law und Advisory) unterteilt. Im Mittelpunkt von Audit steht die Prüfung von Jahres- und Konzernabschlüssen. Tax steht für die steuerberatende¬ und Law für die rechtsberatende Tätigkeit von KPMG. Der Bereich Advisory bündelt das fachliche Know-how zu betriebswirtschaftlichen, regulatorischen und transaktionsorientierten Themen.\nwww.kpmg.at",
        "SAP und andere hier erwähnte SAP-Produkte und -Dienstleistungen sowie deren jeweilige Logos sind Marken oder eingetragene Marken der SAP SE in Deutschland und anderen Ländern. Bitte besuchen Sie https://www.sap.com/corporate/en/legal/copyright.html für zusätzliche Informationen und Hinweise zur Trademark. Alle anderen genannten Produkt- und Dienstleistungsnamen sind Marken ihrer jeweiligen Unternehmen.",
        "Weitere Informationen sowie Fotomaterial zum Download finden Sie hier: KPMG Newsroom- \"KPMG Österreich tritt SAP-PartnerEdge-Programm bei\"",
        "Rückfragen & Kontakt\nKPMG Austria GmbH\nMag. Lisa Kannonier\nPorzellangasse 51\n1090 Wien\n+43 664 8213 664\nlkannonier@kpmg.at"
    ]
    for statement in unwanted_statements:
        text = text.replace(statement, "").strip()
    return text


def detect_abbreviations(text):
    """Detect abbreviations in the text using a predefined list and regex."""
    # Common German abbreviations
    abbreviations = ["z.B.", "etc.", "u.a.", "d.h.", "i.e.", "e.g.", "bzw.", "ca.", "usw.", "v.a.", "Nr.", "bspw.", "ggf."]
    found_abbreviations = [abbr for abbr in abbreviations if abbr in text]

    # Regex to detect groups of capital letters or dot-stopped abbreviations
    regex_abbreviations = re.findall(r'\b[A-ZÄÖÜ]{2,}\b|\b(?:[A-Za-z]{1,3}\.){2,}', text)

    all_abbreviations = list(set(found_abbreviations + regex_abbreviations))
    return len(all_abbreviations), all_abbreviations[:3]

def detect_accumulations(text):
    """
    Detect accumulations in the text by identifying sequences of related words.
    """
    accumulations = []
    
    # Regex pattern to find comma-separated or conjunction-separated words
    pattern = r'((\b\w+\b)(,\s*\b\w+\b)+|(\b\w+\b\s+(und|oder)\s+\b\w+\b))'
    matches = re.findall(pattern, text)

    for match in matches:
        group = match[0]
        # Tokenize and check if the words belong to a common POS group
        tokens = word_tokenize(group)
        pos_tags = [tag for _, tag in pos_tag(tokens) if tag.startswith(('NN', 'JJ', 'VB'))]
        if len(pos_tags) >= 2:  # Ensure at least two descriptive or related words
            accumulations.append(group)

    return len(accumulations), accumulations[:5]  # Limit examples to 5 for readability


def detect_alliterations(text):
    """
    Detect alliterations in the text by identifying consecutive words starting with the same letter.
    """
    tokens = word_tokenize(text)  # Tokenize the text
    alliterations = []
    for i in range(len(tokens) - 1):
        # Check if consecutive words start with the same letter and are alphabetic
        if tokens[i][0].lower() == tokens[i + 1][0].lower() and tokens[i].isalpha() and tokens[i + 1].isalpha():
            alliterations.append(f"{tokens[i]} {tokens[i + 1]}")
    return len(alliterations), list(set(alliterations))  # Return count and unique examples


    for match in matches:
        group = match[0]
        # Tokenize and check if the words belong to a common POS group
        tokens = word_tokenize(group)
        pos_tags = [tag for _, tag in pos_tag(tokens) if tag.startswith(('NN', 'JJ', 'VB'))]
        if len(pos_tags) >= 2:  # Ensure at least two descriptive or related words
            accumulations.append(group)

    return len(accumulations), accumulations[:5]  # Limit examples to 5 for readability


def detect_allusions(text):
    """
    Use NER to identify potential allusions in the text.
    """
    doc = nlp(text)
    # Extract entities related to people, locations, events, organizations, or works of art
    entities = [ent.text for ent in doc.ents if ent.label_ in {"PERSON", "ORG", "GPE", "EVENT", "WORK_OF_ART"}]
    return len(entities), entities[:5]  # Return count and up to 5 examples

def detect_anacoluthon(text):
    """
    Detect potential anacoluthons in a text.
    An anacoluthon is a break in grammatical structure, often indicated by incomplete clauses or shifts in syntax.
    """
    # Heuristic pattern to detect sudden interruptions or incomplete clauses
    pattern = r'(\baber\b|\bund\b|\boder\b)[^.,!?]*?[.!?]\s*[A-ZÄÖÜ]'
    
    matches = re.findall(pattern, text)
    return len(matches), matches[:5]  # Return count and up to 5 examples

def detect_anaphora(text):
    """Detect anaphora by identifying repeated words or phrases at the start of statements."""
    # Split text into statements separated by , . ; "und" "aber" "oder"
    statements = re.split(r'[.,;]|\b(?:und|aber|oder)\b', text)
    anaphora_instances = []
    previous_start = None

    for statement in statements:
        statement = statement.strip()
        if not statement:
            continue
        words = word_tokenize(statement)
        if words:
            current_start = words[0].lower()
            if previous_start == current_start:
                anaphora_instances.append(statement)
            previous_start = current_start

    return len(anaphora_instances), anaphora_instances[:5]  # Return count and examples

def detect_anglicisms(text):
    """
    Detect anglicisms in the text using SpaCy tokenization and language identification.
    """
    doc = nlp(text)
    anglicisms = []

    # Check tokens for foreign words, heuristically focusing on English
    for token in doc:
        if token.is_alpha and token.lang_ == 'en':  # Token identified as English
            anglicisms.append(token.text)

    # Fallback: Check for common anglicisms explicitly
    predefined_anglicisms = [
        "team", "marketing", "feedback", "update", "start-up", "meeting",
        "business", "design", "performance", "download", "workshop", "job"
    ]
    anglicisms += [word for word in text.split() if word.lower() in predefined_anglicisms]

    return len(set(anglicisms)), list(set(anglicisms))[:5]  # Return unique count and examples

def detect_antimetabole(text):
    """
    Detect antimetabole by finding phrases or words repeated in reverse order.
    """
    antimetabole_instances = []
    sentences = re.split(r'[.!?]', text)  # Split text into sentences

    for sentence in sentences:
        words = word_tokenize(sentence.strip().lower())  # Tokenize the sentence
        if len(words) < 4:  # Skip very short sentences
            continue

        # Check for repeated phrases in reverse order
        for i in range(len(words) - 1):
            for j in range(i + 2, len(words) + 1):
                phrase = words[i:j]
                reversed_phrase = list(reversed(phrase))
                if reversed_phrase in [words[k:k+len(phrase)] for k in range(len(words) - len(phrase) + 1)]:
                    antimetabole_instances.append(" ".join(phrase))

    return len(antimetabole_instances), list(set(antimetabole_instances))[:5]  # Return count and examples

def detect_antithesis(text):
    """
    Detect antithesis by identifying sentences with contrasting ideas.
    Looks for specific contrast conjunctions and patterns in the sentence.
    """
    antithesis_instances = []
    sentences = re.split(r'[.!?]', text)  # Split text into sentences

    # Common German contrast conjunctions
    contrast_conjunctions = [
        "aber", "doch", "jedoch", "sondern", "während", "hingegen", "im Gegensatz zu", "dagegen", "obwohl", "trotzdem"
    ]

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue

        # Check for conjunctions indicating contrast
        for conj in contrast_conjunctions:
            # Ensure the sentence contains a conjunction and has a complex structure
            if conj in sentence and re.search(r'\w+.*?'+re.escape(conj)+r'.*?\w+', sentence):
                antithesis_instances.append(sentence)
                break

    return len(antithesis_instances), antithesis_instances[:5]  # Return count and examples

archaic_words = [
    "allda", "allhie", "allzumal", "allzeit", "alsbald", "alsdann", "alsogleich",
    "anheim", "baldens", "balde", "bannen", "darob", "darum", "dieweil",
    "ehedem", "ehern", "einstmals", "entbehren", "entfleuchen", "erachtens",
    "erhaben", "erquicken", "erstlich", "fernab", "fernerhin", "frohen",
    "garstig", "geziemen", "geziemend", "gleichermaßen", "hehr", "hernach",
    "hinfort", "hinieden", "hinweg", "hochwohlgeboren", "hochwürden",
    "immerdar", "jeher", "jedweder", "jener", "künftig", "leidenschaftslos",
    "lieblich", "liederlich", "löblich", "manigfaltig", "mildtätig", "mühselig",
    "mündig", "nimmermehr", "nochmals", "obgleich", "obschon", "obzwar",
    "ohnedies", "ohnfehlbar", "ohnmächtig", "ohngefähr", "ohnschädlich",
    "prahlen", "redlich", "reizend", "schier", "schmählich", "schnellstens",
    "sogleich", "sonach", "stattlich", "stets", "traun", "tunlich", "überdies",
    "übermütig", "ungefährlich", "unverzüglich", "unzweifelhaft", "unbillig",
    "unverbrüchlich", "untertan", "unvermindert", "vielmals", "weiland",
    "wehklagen", "wohlan", "wohlgefällig", "wundersam", "zumeist", "zusehends",
    "zuvörderst", "zuweilen", "zwiespältig", "zwölfmal"
]

def detect_archaisms(text):
    """
    Detect archaisms in the text using a predefined list and SpaCy tokenization.
    """
    doc = nlp(text)
    found_archaisms = [token.text for token in doc if token.text.lower() in archaic_words]

    return len(found_archaisms), list(set(found_archaisms))[:5]  # Return count and examples

def detect_assonance(text):
    """
    Detect assonance in the text by identifying consecutive words with shared vowel patterns.
    A higher threshold ensures at least three shared vowels between consecutive words.
    """
    vowels = "aeiouäöüAEIOUÄÖÜ"
    tokens = word_tokenize(text)  # Tokenize the text
    tokens = [token for token in tokens if token.isalpha()]  # Keep only alphabetic tokens

    assonance_instances = []
    for i in range(len(tokens) - 1):
        # Extract vowels from consecutive words
        vowels_in_word1 = set([char.lower() for char in tokens[i] if char in vowels])
        vowels_in_word2 = set([char.lower() for char in tokens[i + 1] if char in vowels])

        # Require at least 3 shared vowels for meaningful assonance
        if len(vowels_in_word1.intersection(vowels_in_word2)) >= 3:
            assonance_instances.append(f"{tokens[i]} {tokens[i + 1]}")

    return len(assonance_instances), list(set(assonance_instances))[:5]  # Return count and examples

def detect_call_to_action(text):
    """
    Detects calls to action (CTAs) in the text using SpaCy for linguistic analysis.
    Focuses on imperative verbs and predefined CTA phrases.
    """
    # Predefined CTA phrases commonly used in German
    cta_phrases = [
        "Jetzt bestellen", "Kontaktieren Sie uns", "Klicken Sie hier", "Melden Sie sich an",
        "Rufen Sie an", "Schreiben Sie uns", "Besuchen Sie unsere Website",
        "Fordern Sie ein Angebot an", "Laden Sie die Broschüre herunter",
        "Erfahren Sie mehr", "Sichern Sie sich", "Holen Sie sich"
    ]

    # Use SpaCy to analyze the text
    doc = nlp(text)

    # Find predefined CTA phrases in the text
    found_phrases = [phrase for phrase in cta_phrases if phrase.lower() in text.lower()]

    # Detect imperative verbs (VERB in imperative mood)
    imperative_verbs = [token.text for token in doc if token.pos_ == "VERB" and token.morph.get("Mood") == ["Imp"]]

    # Combine predefined phrases and imperative verbs as CTA instances
    cta_instances = list(set(found_phrases + imperative_verbs))

    return len(cta_instances), cta_instances[:5]  # Return count and up to 5 examples

def detect_comparisons(text):
    """
    Detect comparisons in the text using SpaCy and predefined comparison keywords.
    """
    comparison_keywords = ["wie", "als", "so", "so ... wie", "im Vergleich zu", "mehr", "weniger", "besser", "schlechter", "größer", "kleiner"]
    doc = nlp(text)
    comparisons = []

    for sentence in doc.sents:
        tokens = [token.text.lower() for token in sentence]
        if any(keyword in tokens for keyword in comparison_keywords):
            comparisons.append(sentence.text)

    return len(comparisons), comparisons[:5]  # Return count and up to 5 examples

def detect_enumerations(text):
    """
    Detect enumerations in the text using SpaCy.
    Enumerations are identified as comma-separated or conjunction-separated lists of items.
    """
    doc = nlp(text)
    enumerations = []

    for sent in doc.sents:
        # Find tokens that are part of enumerations
        tokens = [token.text for token in sent]
        if "," in tokens or "und" in tokens or "oder" in tokens:
            matches = re.findall(r'\b(\w+(?:,\s*\w+)*\s*(?:und|oder)\s*\w+)\b', sent.text)
            enumerations.extend(matches)

    return len(enumerations), list(set(enumerations))[:5]  # Return count and up to 5 unique examples

def detect_epithets(text):
    """
    Detect epithets in the text using SpaCy.
    Epithets are descriptive adjectives or phrases used to characterize a person or thing.
    """
    doc = nlp(text)
    epithets = []

    # Check for adjectives (amplified by nouns or proper nouns) that are descriptive
    for token in doc:
        if token.pos_ == "ADJ" and token.head.pos_ in {"NOUN", "PROPN"}:
            # Check if the adjective describes the noun/proper noun
            epithets.append(f"{token.text} {token.head.text}")

    return len(epithets), list(set(epithets))[:5]  # Return count and examples

def detect_etymological_appeal(text):
    """
    Detect words or phrases in the text that indicate an etymological appeal.
    Focus on mentions of language roots, historical origins, or classical references.
    """
    # Predefined list of common etymological keywords and phrases
    etymology_keywords = [
        "latein", "altgriechisch", "griechisch", "latinisierung", "stammt aus",
        "ursprung", "abgeleitet von", "bedeutet", "ursprünglich", "wurzel",
        "herkunft", "etymologie", "sprachgeschichte", "altenglisch", "germanisch",
        "romanisch", "indogermanisch", "slawisch", "gotisch", "historisch",
        "archaik", "lehnwort", "sprachlich", "dialekt", "idiom", "wurde entlehnt"
    ]

    # Tokenize and process the text using SpaCy
    doc = nlp(text)

    # Detect matches based on keywords
    matches = [token.text for token in doc if any(keyword in token.text.lower() for keyword in etymology_keywords)]

    # Return the count and examples
    return len(matches), matches[:5]

def detect_gendered_language(text):
    """
    Detect gendered language in German text using SpaCy.
    Focuses on patterns such as words ending with "innen" or other gender-specific suffixes.
    """
    # Load German NLP model
    nlp = spacy.load("de_core_news_sm")

    # Define gendered suffix patterns
    gendered_suffixes = ["innen", "*in", "*innen"]

    # Process the text
    doc = nlp(text)

    # Detect words ending with gendered suffixes
    gendered_words = [
        token.text for token in doc
        if any(token.text.endswith(suffix) for suffix in gendered_suffixes)
    ]

    return len(gendered_words), list(set(gendered_words))[:5]  # Return count and examples


def detect_hyperbaton(text):
    """
    Detects hyperbaton by identifying unusual word orders in sentences, 
    focusing on separated adjective-noun pairs or split clauses.
    """
    doc = nlp(text)
    hyperbaton_instances = []

    for sent in doc.sents:
        adjectives = [token for token in sent if token.pos_ == "ADJ"]
        nouns = [token for token in sent if token.pos_ == "NOUN"]

        # Check if adjectives and nouns are separated by more than 3 words
        for adj in adjectives:
            for noun in nouns:
                if adj.i < noun.i and 0 < (noun.i - adj.i) <= 3:
                    hyperbaton_instances.append(sent.text)

    return len(hyperbaton_instances), hyperbaton_instances[:5]  # Limit examples to 5


def detect_hyperbole(text):
    """
    Detects hyperboles by searching for exaggerative words or phrases in the text.
    Uses predefined lists of hyperbolic expressions common in German.
    """
    hyperbole_keywords = [
        "unendlich", "immer", "ewig", "absolut", "gigantisch", "riesig", 
        "unvorstellbar", "unfassbar", "unerreichbar", "total", "mega", "extrem"
    ]

    doc = nlp(text)
    hyperbole_instances = [token.text for token in doc if token.text.lower() in hyperbole_keywords]

    return len(hyperbole_instances), list(set(hyperbole_instances))[:5]  # Return count and unique examples

def detect_hypotaxis(text):
    """
    Detects hypotaxis by identifying sentences with a high density of subordinating conjunctions.
    """
    subordinating_conjunctions = [
        "weil", "obwohl", "als", "nachdem", "bevor", "während", 
        "damit", "sobald", "sowie", "indem", "falls", "wenn", "ob"
    ]

    doc = nlp(text)
    hypotaxis_instances = []

    for sent in doc.sents:
        conjunctions = [token for token in sent if token.text.lower() in subordinating_conjunctions]
        if len(conjunctions) >= 2:  # Consider sentences with at least 2 conjunctions as hypotactic
            hypotaxis_instances.append(sent.text)

    return len(hypotaxis_instances), hypotaxis_instances[:5]  # Limit examples to 5


def detect_intertextuality(text):
    """
    Detects intertextuality by identifying references to other texts, works, or events using named entities.
    """
    doc = nlp(text)
    intertextuality_instances = []

    # Extract entities related to works of art, events, or organizations
    for ent in doc.ents:
        if ent.label_ in {"WORK_OF_ART", "EVENT", "ORG", "PERSON"}:
            intertextuality_instances.append(ent.text)

    return len(intertextuality_instances), list(set(intertextuality_instances))[:5]  # Return count and unique examples

def detect_latinisms(text):
    """
    Detects Latinisms by identifying Latin-origin words and phrases commonly used in German texts.
    """
    latinisms = [
    "a posteriori", "a priori", "ad absurdum", "ad acta", "ad hoc", "ad infinitum", "ad interim", "ad libitum",
    "ad valorem", "alias", "alibi", "alter ego", "annus horribilis", "annus mirabilis", "aqua", "bona fide",
    "carpe diem", "casus belli", "ceteris paribus", "circa", "cogito ergo sum", "compos mentis", "contra",
    "corpus", "corrigendum", "curriculum vitae", "de facto", "de iure", "de novo", "deus ex machina", "dictum",
    "ergo", "et al.", "et cetera", "et seq.", "et tu, Brute?", "ex aequo", "ex cathedra", "ex libris", "ex nihilo",
    "ex officio", "ex post facto", "extra muros", "fac simile", "falsum", "festina lente", "fiat", "ibidem", "idem",
    "id est (i.e.)", "in absentia", "in aeternum", "in camera", "in dubio pro reo", "in extenso", "in flagranti",
    "in loco parentis", "in medias res", "in memoriam", "in nuce", "in pleno", "in situ", "in spe", "in toto",
    "ipso facto", "lapsus", "lex talionis", "loco citato", "locus classicus", "magnum opus", "mea culpa",
    "modus operandi", "modus vivendi", "mutatis mutandis", "ne plus ultra", "nomen est omen", "non compos mentis",
    "non sequitur", "nota bene", "nulla poena sine lege", "opus", "opus magnum", "pax", "per aspera ad astra",
    "per capita", "per diem", "per se", "persona non grata", "post hoc", "post scriptum (P.S.)", "prima facie",
    "pro bono", "pro forma", "pro rata", "pro tempore", "quid pro quo", "quod erat demonstrandum (Q.E.D.)",
    "sic", "sic transit gloria mundi", "sine qua non", "status quo", "sub rosa", "sub verbo", "sui generis",
    "tabula rasa", "tempus fugit", "terra incognita", "ultima ratio", "verbatim", "versus", "vice versa", "viva voce",
    "vox populi", "veni, vidi, vici", "vade mecum", "ad nauseam", "in vino veritas", "amor fati", "carthago delenda est",
    "ubi dubium ibi libertas", "alea iacta est", "amor vincit omnia", "in dubio pro libertate", "fiat lux", "lux in tenebris",
    "felix culpa", "memento mori", "ars longa, vita brevis", "veni, vidi, vici", "contra legem", "in absentia",
    "pacta sunt servanda", "aurea mediocritas", "noli me tangere", "dum spiro, spero", "cogito, ergo sum",
    "ad hominem", "de gustibus non est disputandum", "pecunia non olet", "divide et impera", "do ut des",
    "homo homini lupus", "nemo me impune lacessit", "festina lente", "salus publica suprema lex esto"
]


    doc = nlp(text)
    found_latinisms = [token.text for token in doc if token.text.lower() in latinisms]

    return len(found_latinisms), list(set(found_latinisms))[:5]  # Return count and unique examples

def detect_litotes(text):
    """
    Detects litotes in German text using SpaCy.
    Looks for negation words followed by positive or neutral descriptors.
    """
    doc = nlp(text)
    litotes = []

    # List of negation words in German
    negations = ["nicht", "kein", "keine", "keiner", "keines", "weder", "noch"]

    # Look for patterns of negation followed by adjectives or adverbs
    for token in doc:
        if token.text.lower() in negations and token.head.pos_ in {"ADJ", "ADV"}:
            phrase = f"{token.text} {token.head.text}"
            litotes.append(phrase)

    return len(litotes), list(set(litotes))[:5]  # Return count and examples

def detect_paradoxes(text):
    """
    Detects paradoxes in German text using SpaCy.
    Looks for sentences with contradictory terms or patterns.
    """
    doc = nlp(text)
    paradox_instances = []

    # Extensive list of contradictory pairs in German
    contradictory_pairs = [
        ("weniger", "mehr"), ("tot", "lebendig"), ("still", "laut"),
        ("schnell", "langsam"), ("hell", "dunkel"), ("arm", "reich"),
        ("schwer", "leicht"), ("heiß", "kalt"), ("jung", "alt"),
        ("oben", "unten"), ("drinnen", "draußen"), ("stark", "schwach"),
        ("frech", "höflich"), ("klug", "dumm"), ("leise", "laut"),
        ("voll", "leer"), ("nass", "trocken"), ("gut", "schlecht"),
        ("neu", "alt"), ("alt", "jung"), ("groß", "klein"),
        ("nah", "fern"), ("freundlich", "feindlich"), ("klar", "unklar"),
        ("glücklich", "traurig"), ("auf", "ab"), ("schön", "hässlich"),
        ("hart", "weich"), ("dick", "dünn"), ("rein", "unrein"),
        ("aktiv", "passiv"), ("heiter", "düster"), ("geboren", "gestorben")
    ]

    for sent in doc.sents:  # Analyze sentence by sentence
        sentence = sent.text.lower()
        for word1, word2 in contradictory_pairs:
            # Check if both words are in the same sentence
            if word1 in sentence and word2 in sentence:
                paradox_instances.append(sentence)

    return len(paradox_instances), list(set(paradox_instances))[:10]  # Return count and up to 10 examples

def detect_periphrases(text):
    """
    Detects periphrases in German text using SpaCy.
    Periphrases are verbose descriptions of simple concepts or ideas.
    """
    doc = nlp(text)
    periphrases = []

    # Common patterns or keywords that indicate periphrases
    verbose_keywords = [
        "von großer Bedeutung", "nicht unbedeutend", "im Rahmen von",
        "im Zuge von", "aufgrund der Tatsache", "mit Blick auf",
        "in der Lage sein", "nicht in der Lage", "dasjenige",
        "in Anbetracht der", "unter Berücksichtigung von",
        "eine Vielzahl von", "es ist zu beachten, dass",
        "es ist erwähnenswert, dass", "auf eine Art und Weise",
        "nicht ohne", "in gewisser Weise", "von entscheidender Bedeutung"
    ]

    # Detect predefined verbose patterns
    for phrase in verbose_keywords:
        if phrase.lower() in text.lower():
            periphrases.append(phrase)

    # Use linguistic patterns to detect complex noun phrases
    for sent in doc.sents:
        for chunk in sent.noun_chunks:
            # Look for long noun phrases with prepositional modifiers
            if len(chunk) > 4 and any(token.dep_ == "prep" for token in chunk):
                periphrases.append(chunk.text)

    return len(set(periphrases)), list(set(periphrases))[:10]  # Return count and up to 10 examples

def detect_personification(text):
    """
    Detects personification in the text using SpaCy.
    Personification is identified when non-human entities are associated with human actions or emotions.
    """
    doc = nlp(text)
    personifications = []

    # Iterate through sentences to detect personification patterns
    for sent in doc.sents:
        for token in sent:
            if token.dep_ in ("nsubj", "nsubj:pass") and token.ent_type_ not in ["PERSON", "ORG"]:
                # Check if the subject is a non-human entity performing a human-like action
                for child in token.children:
                    if child.pos_ == "VERB" or child.pos_ == "ADJ":  # Verb or descriptive adjective
                        personifications.append(f"{token.text} {child.text}")

    return len(set(personifications)), list(set(personifications))[:10]  # Return count and examples

def detect_rhetorical_questions(text):
    """
    Detect rhetorical questions in the text using SpaCy.
    Focuses on sentences with question marks but statements in their structure.
    """
    doc = nlp(text)
    rhetorical_questions = []

    for sent in doc.sents:
        if "?" in sent.text:
            # Look for rhetorical patterns (e.g., negations, obvious answers)
            if any(word.text.lower() in ["nicht", "doch", "wirklich", "etwa"] for word in sent):
                rhetorical_questions.append(sent.text)

    return len(rhetorical_questions), rhetorical_questions[:5]  # Return count and examples

def detect_temporal_anchoring(text):
    """
    Detect temporal anchoring by identifying references to time.
    Uses SpaCy NER for dates, times, and durations.
    """
    doc = nlp(text)
    temporal_anchors = [ent.text for ent in doc.ents if ent.label_ in {"DATE", "TIME", "DURATION"}]

    return len(temporal_anchors), temporal_anchors[:5]  # Return count and examples

def detect_tone_of_voice(text):
    """
    Detect the tone of voice in the text using SpaCy.
    Focuses on adjectives, adverbs, and modal verbs.
    """
    doc = nlp(text)
    tone_indicators = []

    for token in doc:
        if token.pos_ in ["ADJ", "ADV"]:  # Adjectives and adverbs
            tone_indicators.append(token.text)
        elif token.pos_ == "VERB" and token.morph.get("Mood") == ["Imp"]:  # Imperative verbs
            tone_indicators.append(token.text)

    return len(tone_indicators), tone_indicators[:5]  # Return count and examples

def detect_word_fields(text):
    """
    Detect word fields/groups in the text using SpaCy similarity and context analysis.
    Groups words with high similarity scores.
    """
    doc = nlp(text)
    word_fields = {}

    for token in doc:
        if token.is_alpha and not token.is_stop:
            for other_token in doc:
                if other_token != token and other_token.is_alpha and not other_token.is_stop:
                    similarity = token.similarity(other_token)
                    if similarity > 0.8:  # Threshold for similarity
                        word_fields.setdefault(token.text, []).append(other_token.text)

    # Format as unique groups
    word_fields = {key: list(set(values)) for key, values in word_fields.items()}
    return len(word_fields), list(word_fields.items())[:5]  # Return count and example groups





results = []
all_filtered_words = []

for file_name in os.listdir(folder_path):
    if file_name.endswith('.txt'):
        file_path = os.path.join(folder_path, file_name)

        with open(file_path, 'r', encoding='utf-8') as file:
            text = file.read()
        
        print("alive")
        # Clean the text by removing the unwanted statement
        text = clean_text(text)

        # Split text into sentences and words
        sentences = text.split('.')
        words = text.split()

        # Calculate metrics
        total_characters = len(text)
        total_words = len(words)
        total_sentences = len([s for s in sentences if s.strip()])
        avg_sentence_length = total_words / total_sentences if total_sentences > 0 else 0

        # Count punctuation marks
        num_dots = text.count('.')
        num_exclamation_marks = text.count('!')
        num_question_marks = text.count('?')

        # POS tagging
        tokens = word_tokenize(text)
        pos_counts = Counter(tag for word, tag in pos_tag(tokens))

        # Using SpaCy for POS tagging
        doc = nlp(text)  # 'nlp' is the SpaCy German model
        pos_counts = Counter(token.pos_ for token in doc)

        # Extract specific POS counts for nouns, adjectives, and verbs
        num_nouns = sum(pos_counts[tag] for tag in ['NOUN', 'PROPN'])  # Noun and Proper Noun
        num_adjectives = sum(pos_counts[tag] for tag in ['ADJ'])  # Adjectives
        num_verbs = sum(pos_counts[tag] for tag in ['VERB'])  # Verbs
        # Frequent words excluding stopwords
        filtered_words = [word for word in tokens if word.lower() not in german_stopwords and word.isalpha()]
        all_filtered_words.extend(filtered_words)  # Add to cumulative list
        word_frequencies = Counter(filtered_words).most_common(10)
        frequent_words = {f"word_{i+1}": word for i, (word, _) in enumerate(word_frequencies)}

        #  detection
        abbreviation_count, abbreviation_examples = detect_abbreviations(text)
        accumulation_count, accumulation_examples = detect_accumulations(text)
        alliteration_count, alliteration_examples = detect_alliterations(text)
        allusion_count, allusion_examples = detect_allusions(text)
        anacoluthon_count, anacoluthon_examples = detect_anacoluthon(text)
        anaphora_count, anaphora_examples = detect_anaphora(text)
        anglicism_count, anglicism_examples = detect_anglicisms(text)
        antimetabole_count, antimetabole_examples = detect_antimetabole(text)
        antithesis_count, antithesis_examples = detect_antithesis(text)
        archaism_count, archaism_examples = detect_archaisms(text)
        assonance_count, assonance_examples = detect_assonance(text)
        cta_count, cta_examples = detect_call_to_action(text)
        comparison_count, comparison_examples = detect_comparisons(text)
        enumeration_count, enumeration_examples = detect_enumerations(text)
        epithet_count, epithet_examples = detect_epithets(text)
        etymology_count, etymology_examples = detect_etymological_appeal(text)
        gendered_count, gendered_examples = detect_gendered_language(text)
        hyperbaton_count, hyperbaton_examples = detect_hyperbaton(text)
        hyperbole_count, hyperbole_examples = detect_hyperbole(text)
        hypotaxis_count, hypotaxis_examples = detect_hypotaxis(text)
        intertextuality_count, intertextuality_examples = detect_intertextuality(text)
        latinism_count, latinism_examples = detect_latinisms(text)
        litotes_count, litotes_examples = detect_litotes(text)
        paradox_count, paradox_examples = detect_paradoxes(text)
        periphrases_count, periphrases_examples = detect_periphrases(text)
        personification_count, personification_examples = detect_personification(text)
        rhetorical_count, rhetorical_examples = detect_rhetorical_questions(text)
        temporal_count, temporal_examples = detect_temporal_anchoring(text)
        tone_count, tone_examples = detect_tone_of_voice(text)
        word_field_count, word_field_examples = detect_word_fields(text)






        # Append results
        result = {
            'file_name': file_name,
            'total_characters': total_characters,
            'total_words': total_words,
            'total_sentences': total_sentences,
            'avg_sentence_length': avg_sentence_length,
            'num_dots': num_dots,
            'num_exclamation_marks': num_exclamation_marks,
            'num_question_marks': num_question_marks,
            'num_nouns': num_nouns,
            'num_adjectives': num_adjectives,
            'num_verbs': num_verbs,
            'abbreviation_count': abbreviation_count,
            'abbreviation_examples': "; ".join(abbreviation_examples),
            'accumulation_count': accumulation_count,
            'accumulation_examples': "; ".join(accumulation_examples),
            'alliteration_count': alliteration_count,
            'alliteration_examples': "; ".join(alliteration_examples),
            'allusion_count': allusion_count,
            'allusion_examples': "; ".join(allusion_examples),
            'anacoluthon_count': anacoluthon_count,
            'anacoluthon_examples': "; ".join(anacoluthon_examples),
            'anaphora_count': anaphora_count,
            'anaphora_examples': "; ".join(anaphora_examples),
            'anglicism_count': anglicism_count,
            'anglicism_examples': "; ".join(anglicism_examples),
            'antimetabole_count': antimetabole_count,
            'antimetabole_examples': "; ".join(antimetabole_examples),
            'antithesis_count': antithesis_count,
            'antithesis_examples': "; ".join(antithesis_examples),
            'archaism_count': archaism_count,
            'archaism_examples': "; ".join(archaism_examples),
            'assonance_count': assonance_count,
            'assonance_examples': "; ".join(assonance_examples),
            'cta_count': cta_count,
            'cta_examples': "; ".join(cta_examples),
            'comparison_count': comparison_count,
            'comparison_examples': "; ".join(comparison_examples),
            'enumeration_count': enumeration_count,
            'enumeration_examples': "; ".join(enumeration_examples),
            'epithet_count': epithet_count,
            'epithet_examples': "; ".join(epithet_examples), 
            'etymology_count': etymology_count,
            'etymology_examples': "; ".join(etymology_examples),
            'gendered_language_count': gendered_count,
            'gendered_language_examples': "; ".join(gendered_examples),
            'hyperbaton_count': hyperbaton_count,
            'hyperbaton_examples': "; ".join(hyperbaton_examples),
            'hyperbole_count': hyperbole_count,
            'hyperbole_examples': "; ".join(hyperbole_examples),
            'hypotaxis_count': hypotaxis_count,
            'hypotaxis_examples': "; ".join(hypotaxis_examples),
            'intertextuality_count': intertextuality_count,
            'intertextuality_examples': "; ".join(intertextuality_examples),
            'latinism_count': latinism_count,
            'latinism_examples': "; ".join(latinism_examples),
            'litotes_count': litotes_count,
            'litotes_examples': "; ".join(litotes_examples),
            'paradox_count': paradox_count,
            'paradox_examples': "; ".join(paradox_examples),
            'periphrases_count': periphrases_count,
            'periphrases_examples': "; ".join(periphrases_examples),
            'personification_count': personification_count,
            'personification_examples': "; ".join(personification_examples),
            'rhetorical_count': rhetorical_count,
            'rhetorical_examples': "; ".join(rhetorical_examples),
            'temporal_count': temporal_count,
            'temporal_examples': "; ".join(temporal_examples),
            'tone_count': tone_count,
            'tone_examples': "; ".join(tone_examples),
            'word_field_count': word_field_count,
            'word_field_examples': "; ".join(f"{k}: {', '.join(v)}" for k, v in word_field_examples),
        }
        result.update(frequent_words)
        results.append(result)


# Define expected columns
expected_columns = [
    'file_name', 'total_characters', 'total_words', 'total_sentences',
    'avg_sentence_length', 'num_dots', 'num_exclamation_marks',
    'num_question_marks', 'num_nouns', 'num_adjectives', 'num_verbs',
    'abbreviation_count', 'abbreviation_examples'
] + [f'word_{i+1}' for i in range(10)]
expected_columns.extend(['accumulation_count', 'accumulation_examples'])
expected_columns.extend(['alliteration_count', 'alliteration_examples'])
expected_columns.extend(['allusion_count', 'allusion_examples'])
expected_columns.extend(['anacoluthon_count', 'anacoluthon_examples'])
expected_columns.extend(['anaphora_count', 'anaphora_examples'])
expected_columns.extend(['anglicism_count', 'anglicism_examples'])
expected_columns.extend(['antimetabole_count', 'antimetabole_examples'])
expected_columns.extend(['antithesis_count', 'antithesis_examples'])
expected_columns.extend(['archaism_count', 'archaism_examples'])
expected_columns.extend(['assonance_count', 'assonance_examples'])
expected_columns.extend(['climax_count', 'climax_examples'])
expected_columns.extend(['cta_count', 'cta_examples'])
expected_columns.extend(['comparison_count', 'comparison_examples'])
expected_columns.extend(['enumeration_count', 'enumeration_examples'])
expected_columns.extend(['epithet_count', 'epithet_examples'])
expected_columns.extend(['etymology_count', 'etymology_examples'])
expected_columns.extend(['gendered_language_count', 'gendered_language_examples'])
expected_columns.extend(['hyperbaton_count', 'hyperbaton_examples'])
expected_columns.extend(['hyperbole_count', 'hyperbole_examples'])
expected_columns.extend(['hypotaxis_count', 'hypotaxis_examples'])
expected_columns.extend(['intertextuality_count', 'intertextuality_examples'])
expected_columns.extend(['latinism_count', 'latinism_examples'])
expected_columns.extend(['litotes_count', 'litotes_examples'])
expected_columns.extend(['paradox_count', 'paradox_examples'])
expected_columns.extend(['periphrases_count', 'periphrases_examples'])
expected_columns.extend(['personification_count', 'personification_examples'])
expected_columns.extend(['rhetorical_count', 'rhetorical_examples'])
expected_columns.extend(['temporal_count', 'temporal_examples'])
expected_columns.extend(['tone_count', 'tone_examples'])
expected_columns.extend(['word_field_count', 'word_field_examples'])





# Ensure all dictionaries have the same keys
results = [{col: result.get(col, 0) for col in expected_columns} for result in results]

# Convert results to DataFrame
results_df = pd.DataFrame(results)

# Calculate cumulative averages
# Calculate cumulative averages and additional metrics for the cumulative data
cumulative_data = {
    'avg_characters': results_df['total_characters'].mean(),
    'avg_words': results_df['total_words'].mean(),
    'avg_sentences': results_df['total_sentences'].mean(),
    'avg_sentence_length': results_df['avg_sentence_length'].mean(),
    'avg_dots': results_df['num_dots'].mean(),
    'avg_exclamation_marks': results_df['num_exclamation_marks'].mean(),
    'avg_question_marks': results_df['num_question_marks'].mean(),
    'avg_nouns': results_df['num_nouns'].mean(),
    'avg_adjectives': results_df['num_adjectives'].mean(),
    'avg_verbs': results_df['num_verbs'].mean(),
    'avg_abbreviation_count': results_df['abbreviation_count'].mean(),
    'avg_accumulation_count': results_df['accumulation_count'].mean(),
    'avg_alliteration_count': results_df['alliteration_count'].mean(),
    'avg_allusion_count': results_df['allusion_count'].mean(),
    'avg_anacoluthon_count': results_df['anacoluthon_count'].mean(),
    'avg_anaphora_count': results_df['anaphora_count'].mean(),
    'avg_anglicism_count': results_df['anglicism_count'].mean(),
    'avg_antimetabole_count': results_df['antimetabole_count'].mean(),
    'avg_antithesis_count': results_df['antithesis_count'].mean(),
    'avg_archaism_count': results_df['archaism_count'].mean(),
    'avg_assonance_count': results_df['assonance_count'].mean(),
    'avg_climax_count': results_df['climax_count'].mean(),
    'avg_cta_count': results_df['cta_count'].mean(),
    'avg_comparison_count': results_df['comparison_count'].mean(),
    'avg_enumeration_count': results_df['enumeration_count'].mean(),
    'avg_epithet_count': results_df['epithet_count'].mean(),
    'avg_etymology_count': results_df['etymology_count'].mean(),
    'avg_gendered_language_count': results_df['gendered_language_count'].mean(),
    'avg_hyperbaton_count': results_df['hyperbaton_count'].mean(),
    'avg_hyperbole_count': results_df['hyperbole_count'].mean(),
    'avg_hypotaxis_count': results_df['hypotaxis_count'].mean(),
    'avg_intertextuality_count': results_df['intertextuality_count'].mean(),
    'avg_latinism_count': results_df['latinism_count'].mean(),
    'avg_litotes_count': results_df['litotes_count'].mean(),
    'avg_paradox_count': results_df['paradox_count'].mean(),
    'avg_periphrases_count': results_df['periphrases_count'].mean(),
    'avg_personification_count': results_df['personification_count'].mean(),
    'avg_rhetorical_count': results_df['rhetorical_count'].mean(),
    'avg_temporal_count': results_df['temporal_count'].mean(),
    'avg_tone_count': results_df['tone_count'].mean(),
    'avg_word_field_count': results_df['word_field_count'].mean()
}

# Add 10 most common words across all texts
all_word_frequencies = Counter(all_filtered_words).most_common(10)
cumulative_data['most_common_words'] = ', '.join([word for word, _ in all_word_frequencies])




cumulative_df = pd.DataFrame([cumulative_data])

# Add a timestamp to file names to avoid conflicts
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
results_file = f'text_analysis_results_{timestamp}.xlsx'
cumulative_file = f'text_analysis_cumulative_{timestamp}.xlsx'

# Save results to Excel files
results_df.to_excel(results_file, index=False, sheet_name='Detailed Analysis KPMG')
cumulative_df.to_excel(cumulative_file, index=False, sheet_name='Cumulative Averages KPMG')


