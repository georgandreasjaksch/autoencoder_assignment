'''
Retrieve the (labelled) features from Wikipedia using the
WikiDownloader class.
'''
from wikidownloader import WikiDownloader
import pandas as pd
from importlib import reload

def get_features(data_dir='data', max_pages=100, max_categories=5):
    '''
    Retrieves the features (articles) including their label (category) from wikipedia
    and stores them in a dataframe
    :param data_dir: where the data files will be saved
    :param max_pages: maximum number of pages/articles to retrieve within a category
    :param max_categories:  maximum number of categories/labels
    '''

    wd = WikiDownloader()
    page_ids, titles, texts, cats = wd.get_all_toplevel_category_pages(max_pages=max_pages, max_categories=max_categories)

    # set-up dataframe to store the retrieved data and map to columns
    df = pd.DataFrame(
        {'id': page_ids,
         'title': titles,
         'text': texts,
         'category': cats
        })

    # save to file
    df.to_pickle("{0}/Wikipedia_{1}_{2}.pkl".format(data_dir, max_pages, max_categories))