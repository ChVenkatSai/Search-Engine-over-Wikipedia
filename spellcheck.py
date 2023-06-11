# PYSPELLCHECKER

# install spellchecker

from spellchecker import SpellChecker

class SpellCheck():

    def correctQuery(self, query):
        """
		Spelling corrections on each query

		Parameters
		----------
		arg1 : string
			A query from the user or the dataset

		Returns
		-------
		string
			The query after correcting spelling errors in it
		"""
        spell = SpellChecker()

        query = query.split()
        corrected_query = ''

        # find those words that may be misspelt
        misspelled = spell.unknown(query)

        for word_index, word in enumerate(query):
            if word not in misspelled:
                corrected_query += word
            else:
                corrected_query += spell.correction(word) # correcting the misspelt words
            if word_index != len(query) - 1:
                corrected_query += ' '

        return corrected_query
