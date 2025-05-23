import re
import unicodedata
from collections import Counter
from typing import List, Set

from langdetect import detect
from langdetect.lang_detect_exception import LangDetectException
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.tokenize.treebank import TreebankWordDetokenizer
from tqdm import tqdm
import pandas as pd
import jieba
import thulac
import pkuseg
import hanlp
import opencc
from snownlp import SnowNLP 
import jieba.posseg as pseg
# from .Chinese_config import ChineseConfig

class TextPreprocessor:
    """
    Text preprocessor class for cleaning and preprocessing text data.

    Parameters
    ----------
    language : str, optional
        Language of the text data (default is "en").
    remove_stopwords : bool, optional
        Whether to remove stopwords from the text data (default is False).
    lowercase : bool, optional
        Whether to convert text to lowercase (default is True).
    remove_punctuation : bool, optional
        Whether to remove punctuation from the text data (default is True).
    remove_numbers : bool, optional
        Whether to remove numbers from the text data (default is True).
    lemmatize : bool, optional
        Whether to lemmatize words in the text data (default is False).
    stem : bool, optional
        Whether to stem words in the text data (default is False).
    expand_contractions : bool, optional
        Whether to expand contractions in the text data (default is True).
    remove_html_tags : bool, optional
        Whether to remove HTML tags from the text data (default is True).
    remove_special_chars : bool, optional
        Whether to remove special characters from the text data (default is True).
    remove_accents : bool, optional
        Whether to remove accents from the text data (default is True).
    remove_english: bool, optional
        Whether to remove english words from the chinese text data (default is True).
    custom_stopwords : set, optional
        Custom stopwords to remove from the text data (default is []).
    detokenize : bool, optional
        Whether to detokenize the text data (default is False).
    min_word_freq : int, optional
        Minimum word frequency to keep in the text data (default is 2).
    max_word_freq : int, optional
        Maximum word frequency to keep in the text data (default is None).
    min_word_length : int, optional
        Minimum word length to keep in the text data (default is 3).
    max_word_length : int, optional
        Maximum word length to keep in the text data (default is None).
    dictionary : set, optional
        Dictionary of words to keep in the text data (default is []).
    remove_words_with_numbers : bool, optional
        Whether to remove words containing numbers from the text data (default is False).
    remove_words_with_special_chars : bool, optional
        Whether to remove words containing special characters from the text data (default is False). 
    stopwords_path : char, optional
        Chinese only, path to stored Chinese stopwords (default is None)

    """

    def __init__(self, **kwargs):
        self.language = kwargs.get("language", "en")
        self.remove_stopwords = kwargs.get("remove_stopwords", False)
        self.lowercase = kwargs.get("lowercase", True)
        self.remove_punctuation = kwargs.get("remove_punctuation", True)
        self.remove_numbers = kwargs.get("remove_numbers", True)
        self.lemmatize = kwargs.get("lemmatize", False)
        self.stem = kwargs.get("stem", False)
        self.expand_contractions = kwargs.get("expand_contractions", True)
        self.remove_html_tags = kwargs.get("remove_html_tags", True)
        self.remove_special_chars = kwargs.get("remove_special_chars", True)
        self.remove_accents = kwargs.get("remove_accents", True)
        self.remove_english = kwargs.get("remove_english", True)
        self.traditional_simple_convert = kwargs.get("traditional_simple_convert", False)
        self.segmentation_tool = kwargs.get("segmentation_tool", 'jieba')
        self.segmentation_dict = kwargs.get("segmentation_dict", None)
        self.remove_pos = kwargs.get("remove_pos", None)
        self.domain = kwargs.get("domain", "default")
        self.custom_stopwords = (
            set(kwargs.get("custom_stopwords", []))
            if kwargs.get("custom_stopwords")
            else set()
        )
        self.detokenize = kwargs.get("detokenize", False)
        self.min_word_freq = kwargs.get("min_word_freq", 2)
        self.max_word_freq = kwargs.get("max_word_freq", None)
        self.min_word_length = kwargs.get("min_word_length", 3)
        self.max_word_length = kwargs.get("max_word_length", None)
        self.dictionary = set(kwargs.get("dictionary", []))
        self.remove_words_with_numbers = kwargs.get("remove_words_with_numbers", False)
        self.remove_words_with_special_chars = kwargs.get(
            "remove_words_with_special_chars", False)
        self.stopwords_path = kwargs.get("stopwords_path", None)
        

        if self.language == "chinese":                 
            self.stoplist = self.load_stopwords()    
        elif self.language != "en" and self.remove_stopwords:          
            self.stop_words = set(stopwords.words(self.language))
        else:                                                        
            self.stop_words = set(stopwords.words("english"))

        if self.language != "chinese":
            self.stop_words.update(self.custom_stopwords)                

        if self.lemmatize:                                           
            self.lemmatizer = WordNetLemmatizer()

        if self.stem:                                                
            self.stemmer = PorterStemmer()
            
        if self.segmentation_tool == "hanlp":
            if self.remove_pos is not None:
                self.tok = hanlp.load(hanlp.pretrained.tok.COARSE_ELECTRA_SMALL_ZH)
                self.pos = hanlp.load(hanlp.pretrained.pos.CTB9_POS_ELECTRA_SMALL)
            else:
                self.tok = hanlp.load(hanlp.pretrained.tok.COARSE_ELECTRA_SMALL_ZH)
        elif self.segmentation_tool == "thulac":
            if isinstance(self.segmentation_dict, str):
                if self.remove_pos is not None:
                    self.thu = thulac.thulac(seg_only=False, user_dict=self.segmentation_dict)
                else:
                    self.thu = thulac.thulac(seg_only=True, user_dict=self.segmentation_dict)
            else:
                if self.remove_pos is not None:
                    self.thu = thulac.thulac(seg_only=False)
                else:
                    self.thu = thulac.thulac(seg_only=True)

        self.contractions_dict = self._load_contractions()           
        self.word_freq = Counter()                                  


    def load_stopwords(self):
        with open(self.stopwords_path, 'r', encoding='UTF-8') as f:
            stopwords = [line.strip() for line in f]
        return pd.DataFrame({'w': stopwords})
            
    
    def segment_text(self, text, tool='jieba', custom_dict=None, remove_pos=None, domain="default"):
        if remove_pos is None:
            if custom_dict is None:
                custom_words = None
            else:
                if isinstance(custom_dict, str):
                    with open(custom_dict, 'r', encoding='utf-8') as f:
                        custom_words = [line.strip() for line in f if line.strip()]
                else:
                    raise ValueError(f"Please provide a custom dictionary file path")
                
            if tool == 'jieba':
                if isinstance(custom_dict, str):
                    jieba.load_userdict(custom_dict)
                words = list(jieba.cut(text))
            elif tool == 'hanlp':
                tok = self.tok
                if custom_words:
                    tok.dict_combine = set(custom_words)
                words = tok(text)
            elif tool == 'pkuseg':
                if isinstance(custom_dict, str):
                    seg = pkuseg.pkuseg(model_name = domain, user_dict=custom_dict)  
                else:
                    seg = pkuseg.pkuseg(model_name = domain)
                words = seg.cut(text)
            elif tool == 'thulac':
                thu = self.thu
                words = thu.cut(text, text=True).split()
            elif tool == 'snownlp':
                if custom_words:
                    raise ValueError(f"SnowNLP does not support custom dictionaries")
                s = SnowNLP(text)
                words = s.words  
            else:
                raise ValueError(f"Unsupported tokenizer: {tool}. Please choose from ['jieba', 'hanlp', 'pkuseg', 'thulac', 'snownlp']")
            filtered_words = [word for word in words if word not in self.stoplist['w'].tolist() and word != ' ']
        else:
            pos_mapping = {
                "jieba": {
                    "a": ["a", "ad", "ag","an"], #adjective
                    "c":["c"], #conjunction
                    "d":["d","df","dg"], #adverb
                    "e":["e"], #interjection
                    "mq":["m","mg","mq","q"],#numerals and quantifiers
                    "n": ["n", "nr","nrfg","nrt", "ns", "nt", "nz"], #noun
                    "p":["p"],#preposition
                    "r":["r"],#pronoun
                    "u":["u","ud","uj","ul","uv","uz"],#auxiliary word
                    "v": ["v", "vd","vg","vi", "vn","vq"]#verb    
                    },
                "pkuseg": {
                    "a": ["a", "ad","an"], #adjective
                    "c":["c"], #conjunction
                    "d":["d"], #adverb
                    "e":["e"], #interjection
                    "mq":["m","q"],#numerals and quantifiers
                    "n": ["n", "nr","nx", "ns", "nt", "nz"], #noun
                    "p":["p"],#preposition
                    "r":["r"],#pronoun
                    "u":["u"],#auxiliary word
                    "v": ["v", "vd","vn","vx"]#verb    
                    },
                "thulac": {
                    "a": ["a"], #adjective
                    "c":["c"], #conjunction
                    "d":["d"], #adverb
                    "e":["e"], #interjection
                    "mq":["m","mq","q"],#numerals and quantifiers
                    "n": ["n", "np", "ns", "ni", "nz"],#noun
                    "p":["p"],#preposition
                    "r":["r"],#pronoun
                    "u":["u"],#auxiliary word
                    "v": ["v"],#verb
                    },
                "hanlp": {  
                    "a": ["JJ","VA"], #adjective
                    "c":["CC","CS",], #conjunction
                    "d":["AD"], #adverb
                    "e":["IJ"], #interjection
                    "mq":["CD","M","q"],#numerals and quantifiers
                    "n": ["NN", "NR", "NT"],#noun
                    "p":["P"],#preposition
                    "r":["PN"],#pronoun
                    "u":["AS","SP"],#auxiliary word
                    "v": ["VC","VE","VV"],#verb
                    }
                }
            if tool not in ["jieba", "pkuseg", "hanlp", "thulac"]:
                raise ValueError(f"Unsupported tokenizer: {tool}. Please choose from ['jieba', 'pkuseg', 'hanlp', 'thulac']")
            else:
                if remove_pos is None:
                    remove_pos = []
                pos_map = pos_mapping[tool]
                remove_tags = set() 
                for pos in remove_pos:
                    if pos in pos_map:
                        remove_tags.update(pos_map[pos])
                        
                if custom_dict is None:
                    custom_words = None
                else:
                    if isinstance(custom_dict, str):
                        with open(custom_dict, 'r', encoding='utf-8') as f:
                            custom_words = [line.strip() for line in f if line.strip()]
                    else:
                        raise ValueError(f"Please provide a custom dictionary file path")
                    
                if tool == 'jieba':
                    if isinstance(custom_dict, str):
                        jieba.load_userdict(custom_dict)
                    words_with_pos = pseg.cut(text)
                    words_pos = [(word, pos) for word, pos in words_with_pos]
                elif tool == 'pkuseg':
                    if isinstance(custom_dict, str):
                        seg = pkuseg.pkuseg(model_name = domain, user_dict=custom_dict, postag=True)
                    else:
                        seg = pkuseg.pkuseg(model_name = domain, postag=True) 
                    words_pos = seg.cut(text)    
                elif tool == 'hanlp':
                    tok = self.tok
                    pos = self.pos
                    if custom_words:
                        tok.dict_combine = set(custom_words)
                    words = tok(text)
                    pos_tags = pos(words)
                    words_pos = list(zip(words, pos_tags))
                elif tool == 'thulac':
                    thu = self.thu
                    result = thu.cut(text,  text=True)
                    words_pos = []
                    for item in result.split(): 
                        word, pos = item.split('_') 
                        words_pos.append((word,  pos))
            filtered_words = [word for word, pos in words_pos if pos not in remove_tags and word not in self.stoplist['w'].tolist() and word != ' ']
        return filtered_words
    
    def _load_contractions(self):
        # Load a dictionary of contractions and their expansions
        contractions_dict = {
            "can't": "cannot",
            "won't": "will not",
            "n't": " not",
            "'re": " are",
            "'s": " is",
            "'d": " would",
            "'ll": " will",
            "'t": " not",
            "'ve": " have",
            "'m": " am",
        }
        return contractions_dict

    def _expand_contractions(self, text):                                 
        contractions_pattern = re.compile(                                
            "({})".format("|".join(self.contractions_dict.keys())),
            flags=re.IGNORECASE | re.DOTALL,
        )

        def expand_match(contraction):
            match = contraction.group(0)
            expanded_contraction = self.contractions_dict.get(match.lower())
            return expanded_contraction

        expanded_text = contractions_pattern.sub(expand_match, text)      
        return expanded_text

    def _remove_html_tags(self, text):                                    
        clean = re.compile("<.*?>")
        return re.sub(clean, " ", text)

    def _remove_special_characters(self, text, language):                           
        if language != "chinese":
            return re.sub(r"[^a-zA-Z0-9\s]", " ", text)
        else:
            return re.sub(r'[^\u4e00-\u9fff\d]+', '', text)

    def _is_traditional(self, text):
            """ Simple check if the text contains traditional characters using regex"""
            return bool(re.search(r'[\u4e00-\u9fff\u3400-\u4DBF\u20000-\u2A6DF\u2A700-\u2B73F]', text))

    def _remove_accents(self, text):                                      
        text = unicodedata.normalize("NFD", text)
        text = text.encode("ascii", "ignore")
        return text.decode("utf-8")

    def _clean_text(self, text):

        if self.language != "chinese":
            text = text.strip()                                               
            if self.lowercase:                                                
                text = text.lower()
            if self.expand_contractions:
                text = self._expand_contractions(text)                       
            if self.remove_html_tags:
                text = self._remove_html_tags(text)
            if self.remove_special_chars:
                text = self._remove_special_characters(text, language="en")
            if self.remove_accents:
                text = self._remove_accents(text)
            if self.remove_numbers:
                text = re.sub(r"\d+", " ", text)
            if self.remove_punctuation:
                text = re.sub(r"[^\w\s]", " ", text)
    
            words = word_tokenize(text)                                       
    
            # Update word frequency counter
            self.word_freq.update(words)                                      
    
            if self.remove_stopwords:
                words = [word for word in words if word not in self.stop_words]  
    
            if self.lemmatize:
                words = [self.lemmatizer.lemmatize(word) for word in words]
    
            if self.stem:
                words = [self.stemmer.stem(word) for word in words]
    
            if self.min_word_freq is not None:                                
                words = [
                    word for word in words if self.word_freq[word] >= self.min_word_freq
                ]
    
            if self.max_word_freq is not None:                                
                words = [
                    word for word in words if self.word_freq[word] <= self.max_word_freq
                ]
    
            if self.min_word_length is not None:
                words = [word for word in words if len(
                    word) >= self.min_word_length]
    
            if self.max_word_length is not None:
                words = [word for word in words if len(
                    word) <= self.max_word_length]
    
            if self.dictionary != set():                                    
                words = [word for word in words if word in self.dictionary]
    
            if self.remove_words_with_numbers:
                words = [word for word in words if not any(
                    char.isdigit() for char in word)]
    
            if self.remove_words_with_special_chars:
                words = [word for word in words if not re.search(
                    r"[^a-zA-Z0-9\s]", word)]
    
            if self.detokenize:                                             
                text = TreebankWordDetokenizer().detokenize(words)          
            else:
                text = " ".join(words)                                      
    
            # Remove double spaces
            text = re.sub(r"\s+", " ", text)
        
        else:
            text = text.strip()  
            
            if self.traditional_simple_convert:
                self.cc = opencc.OpenCC('t2s.json')
                if self._is_traditional(text):
                    text = self.cc.convert(text)
                
            if self.remove_html_tags:
                text = self._remove_html_tags(text)
            if self.remove_special_chars:
                text = self._remove_special_characters(text, language="chinese")
            if self.remove_numbers:
                text = re.sub(r"\d+", " ", text)
            if self.remove_punctuation:
                text = re.sub(r"[^\w\s]", " ", text)
            if self.remove_english:
                text = re.sub(r"[a-zA-Z]+", " ", text)

            if self.domain in ["default", "web", "news", "medicine", "tourism"]:
                words = self.segment_text(text, tool=self.segmentation_tool, 
                                          custom_dict=self.segmentation_dict, 
                                          remove_pos=self.remove_pos,
                                          domain = self.domain)
            else:
                raise ValueError(f"Please domain choose from ['default', 'web', 'news', 'medicine', 'tourism']")

            # Update word frequency counter
            self.word_freq.update(words)  

            if self.min_word_freq is not None:  
                words = [
                    word for word in words if self.word_freq[word] >= self.min_word_freq
                ]

            if self.max_word_freq is not None:  
                words = [
                    word for word in words if self.word_freq[word] <= self.max_word_freq
                ]

            if self.min_word_length is not None:
                words = [word for word in words if len(
                    word) >= self.min_word_length]

            if self.max_word_length is not None:
                words = [word for word in words if len(
                    word) <= self.max_word_length]

            if self.dictionary != set():  
                words = [word for word in words if word in self.dictionary]

            if self.remove_words_with_numbers:
                words = [word for word in words if not any(
                    char.isdigit() for char in word)]

            if self.remove_words_with_special_chars:
                words = [word for word in words if not re.search(
                    r'[^\u4e00-\u9fff\d]+', word)]

            if self.detokenize:
                text = "".join(words)
            else:
                text = " ".join(words)
                
            # Remove double spaces
            text = re.sub(r"\s+", " ", text)

        return text
    
    # def detect_language(self, text):
    #     if re.search(r'[\u4e00-\u9fff]', text):
    #         return "chinese"
    #     else:
    #         return "en"
    
    def preprocess_text(self, text):
        """
        Preprocess a single text document.

        Parameters
        ----------
        text : str
            Text document to preprocess.

        Returns
        -------
        str
            Preprocessed text document.

        """
        # try:
        #     language = self.detect_language(text)
        #     if language != self.language:
        #         return text
        # except LangDetectException:
        #     pass
        return self._clean_text(text)                                   

    def preprocess_dataframe(self, df, text_column):
        """
        Preprocess a pandas DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame containing text data.
        text_column : str
            Name of the column containing text data.

        Returns
        -------
        pd.DataFrame
            Preprocessed DataFrame.

        """
        df[text_column] = df[text_column].apply(self.preprocess_text)     
        return df

    def preprocess_documents(self, documents: List[str]) -> List[str]:    
        preprocessed_docs = []
        for doc in tqdm(documents, desc="Preprocessing documents"):       
            preprocessed_docs.append(self.preprocess_text(doc))
        return preprocessed_docs

    def add_custom_stopwords(self, stopwords: Set[str]):      #not implemented  for Chinese yet        
        """
        Add custom stopwords to the preprocessor.

        Parameters
        ----------
        stopwords : set
            Set of custom stopwords to be added.
        """
        self.custom_stopwords.update(stopwords)
        self.stop_words.update(stopwords)

    def remove_custom_stopwords(self, stopwords: Set[str]):      #not implemented  for Chinese yet           
        """
        Remove custom stopwords from the preprocessor.

        Parameters
        ----------
        stopwords : set
            Set of custom stopwords to be removed.
        """
        self.custom_stopwords.difference_update(stopwords)
        self.stop_words.difference_update(stopwords)
