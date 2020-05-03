import wikipediaapi

wiki_wiki = wikipediaapi.Wikipedia('ru')


def add_sections(sections, text, level=0):
    ignored_sections = ["См. также", "Примечания", "Литература", "Ссылки"]
    clean_sections = [s for s in sections if s.title not in ignored_sections]
    for s in clean_sections:
        text += s.text
        add_sections(s.sections, text, level=level + 1)

    return text


def collect_articles(*titles):
    text = ""
    for title in titles:
        page = wiki_wiki.page(title)
        text += page.summary + "\n"
        text += add_sections(page.sections, text) + "\n"

    return text
