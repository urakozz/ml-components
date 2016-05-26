class WordNumericEncoder:
    def __init__(self, words, common=0, rare_word_token="UNK"):
        self._words = words
        self._rare_word_token = rare_word_token
        self._counter = collections.Counter(words)
        
        self._set_items(common)
        self._build_dictionary()
        self._encode_words()
        
    def _set_items(self, common=0):
        self._items = [[self._rare_word_token, -1]]
        if common <= 0:
            common = len(self._words)
            self._items = []
        self._items.extend(self._counter.most_common(common))
    
    def _build_dictionary(self):
        self._dictionary = dict()
        for word, _ in self._items:
            self._dictionary[word] = len(self._dictionary)
    
    def _encode_words(self):
        data = list()
        unk_count = 0
        for word in self._words:
            if word in self._dictionary:
                index = self._dictionary[word]
            else:
                index = 0  # items['UNK']
                unk_count = unk_count + 1
            data.append(index)
        self._items[0][1] = unk_count
        self._data = data
        
    def get_data(self):
        return self._data
    
    def get_reverse_dictionary(self):
        return dict(zip(self._dictionary.values(), self._dictionary.keys())) 
        
        
#e = WordNumericEncoder(words[:20], 15)
#encoded = e.get_data()[:10]
#reverse = e.get_reverse_dictionary()
#print(words[:10]) # ['anarchism', 'originated', 'as', 'a', 'term', 'of', 'abuse', 'first', 'used', 'against']
#print(encoded) # [0, 5, 4, 13, 14, 1, 7, 0, 6, 11]
#print([reverse[i] for i in encoded]) # ['UNK', 'originated', 'as', 'a', 'term', 'of', 'abuse', 'UNK', 'used', 'against']

