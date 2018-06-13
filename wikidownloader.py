import wikipediaapi
import mwparserfromhell as mwparse
import sys
from random import randint
from time import sleep

wiki_wiki = wikipediaapi.Wikipedia('en')


class WikiDownloader:
    '''
    Class to retrieve pages/articles from wikipedia
    '''

    def progbar(self, curr, total, full_progbar):
        '''
        Display a progress bar
        :param curr: current number of items
        :param total: total number of items
        :param full_progbar: length of the progress bar
        '''
        frac = curr / total
        filled_progbar = round(frac * full_progbar)
        print('\r', 'o' * filled_progbar + '-' * (full_progbar - filled_progbar), '[{:>7.2%}]'.format(frac), end='')

    def get_category_pages(self, main_category, category, level=0, max_pages=10, max_level=1, page_ids=[], titles=[],
                           texts=[]):
        '''
        Retrieve pages that are contained in a category, also recursively from the subcategories
        until a certain number of pages is collected and up to a certain depth
        :param main_category: main category in wikipedia
        :param category: current category that is scanned for pages
        :param level: level of depth in category hierarchy
        :param max_pages: number of pages to retrieve
        :param max_level: maximum level to descend into
        :param page_ids: list of page ids
        :param titles: list of page titles
        :param texts: list of page texts
        '''

        # retrieve category page
        category_page = wiki_wiki.page(category)

        # descend into pages/categories
        for categorymember in category_page.categorymembers.values():

            #sleep a random amount of secs to avoid blocking
            #sleep(randint(1, 4))

            # namespace (ns) = 0 means page with content
            if categorymember.ns == 0 and len(page_ids) < max_pages:

                categorymember_page = wiki_wiki.page(categorymember.title)

                if categorymember_page.text:

                    #strip the wiki formatting to retrieve plain text
                    categorymember_page_text = mwparse.parse(categorymember_page.text).strip_code()

                    page_ids.append(categorymember_page.pageid)
                    titles.append(categorymember_page.title)
                    texts.append(categorymember_page_text)
                    self.progbar(len(page_ids), max_pages, 20)
                    sys.stdout.flush()

            # namespace (ns) = 14 means category
            if categorymember.ns == 14 and level <= max_level and len(page_ids) < max_pages:
                self.get_category_pages(main_category, categorymember.title, level + 1, max_pages=max_pages,
                                        page_ids=page_ids, titles=titles, texts=texts)

            if len(page_ids) >= max_pages:
                return page_ids, titles, texts, main_category

        return page_ids, titles, texts, main_category

    def get_toplevel_categories(self):
        '''
        Scans the wikipedia top level categories page and returns the categories
        '''
        main_categories = wiki_wiki.page("Category:Main_topic_classifications")
        return main_categories.categorymembers

    def get_all_toplevel_category_pages(self, max_pages=10, max_categories=10):
        '''
        Retrieves all top level category pages from Wikipedia
        :param max_pages: number of pages to retrieve per category
        :param max_categories: number of categories to retrieve
        '''

        page_ids = []
        titles = []
        texts = []
        cats = []

        # get the top level categories
        top_level_cat = self.get_toplevel_categories()
        
        i = 0

        for category in top_level_cat:
            i += 1
            print("Getting %s" % category)
            # try downloading the category pages (timeouts can happen)
            try:
                cat_page_id, cat_page_title, cat_page_text, cat_page_cat = self.get_category_pages(category, category,
                                                                                            max_pages=max_pages,
                                                                                            page_ids=[], titles=[],
                                                                                            texts=[])
                cat_append = [category] * max_pages
                page_ids += cat_page_id
                titles += cat_page_title
                texts += cat_page_text
                cats += cat_append

                print("")
            except:
                continue

            if i == (max_categories):
                return page_ids, titles, texts, cats

        return page_ids, titles, texts, cats