import json
import wikipedia
pages = json.load(open("wikipedia_concepts_content.json", 'r'))[:]

search_results = wikipedia.search("Aerodynamics")
# print(search)
for item in search_results:
	# print(item)
	page = None
	try:
		# page = wikipedia.summary(item, auto_suggest = False)
		page = wikipedia.page(item, auto_suggest = True)
	except Exception:
		continue

	# except wikipedia.DisambiguationError or wikipedia.PageError:
	# 	continue
	pages.append({
        "title" : page.title,
        "content": page.content
	})

search_results = wikipedia.search("Fluid Dynamics")
# print(search)
for item in search_results:
	# print(item)
	page = None
	try:
		# page = wikipedia.summary(item, auto_suggest = False)
		page = wikipedia.page(item, auto_suggest = True)
	except Exception:
		continue

	# except wikipedia.DisambiguationError or wikipedia.PageError:
	# 	continue
	pages.append({
        "title" : page.title,
        "content": page.content
	})

pages = list({page["title"]:page for page in pages}.values())
print(len(pages))
with open("wikipedia_concepts_content.json", 'w') as fout:
	json.dump(pages, fout)
