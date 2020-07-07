import wikipediaapi

wiki_wiki = wikipediaapi.Wikipedia('ru')


def add_sections(sections, text, level=0, selected_sections=None):
    clean_sections = [s for s in sections if s.title in selected_sections]
    for s in clean_sections:
        text += s.text
        add_sections(s.sections, text, level=level + 1, selected_sections=selected_sections)

    return text


def collect_wiki_articles(titles, selected_sections):
    text = ""
    for title in titles:
        page = wiki_wiki.page(title)
        text += page.summary + "\n"
        text += add_sections(page.sections, text, selected_sections=selected_sections) + "\n"

    return text
