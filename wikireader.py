import wikipediaapi

wiki_wiki = wikipediaapi.Wikipedia('ru')


def add_sections(sections, text, level=0, selected_sections=None):
    if selected_sections == None:
        selected_sections = ["Основные понятия", "Определение ООП и его основные концепции",
                             "Особенности реализации", "Объектно-ориентированные языки"]
    clean_sections = [s for s in sections if s.title in selected_sections]
    for s in clean_sections:
        text += s.text
        add_sections(s.sections, text, level=level + 1)

    return text


def collect_wiki_articles(*titles):
    text = ""
    for title in titles:
        page = wiki_wiki.page(title)
        text += page.summary + "\n"
        text += add_sections(page.sections, text) + "\n"

    return text
