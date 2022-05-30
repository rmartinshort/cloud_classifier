import os
import time
import uuid

import pandas as pd
from selenium import webdriver

from cloud_classifier.cloud_classifier.common.constants import (
    CHROMEDRIVER_PATH
)
from cloud_classifier.cloud_classifier.scrape.image_fetcher import (
    download_image_from_url
)


def fetch_image_urls(query: str,
                     max_links_to_fetch: int,
                     wd: webdriver,
                     sleep_between_interactions: int = 1
                     ):
    """
    Function inspired by
    https://medium.com/@wwwanandsuresh/web-scraping-images-from-google-9084545808a2
    to fetch image urls from google image search
    """

    def scroll_to_end(wd):
        wd.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(sleep_between_interactions)

        # build the google query

    search_url = "https://www.google.com/search?safe=off&site=&tbm=isch&source=hp&q={q}&oq={q}&gs_l=img"

    # load the page
    wd.get(search_url.format(q=query))

    image_urls = set()
    image_count = 0
    results_start = 0

    while image_count < max_links_to_fetch:
        scroll_to_end(wd)

        # get all image thumbnail results
        thumbnail_results = wd.find_elements_by_css_selector("img.Q4LuWd")
        number_results = len(thumbnail_results)

        print(f"Found: {number_results} search results. Extracting links from {results_start}:{number_results}")

        for img in thumbnail_results[results_start:number_results]:
            # try to click every thumbnail such that we can get the real image behind it
            try:
                img.click()
                time.sleep(sleep_between_interactions)
            except Exception:
                continue

            # extract image urls
            actual_images = wd.find_elements_by_css_selector('img.n3VNCb')
            for actual_image in actual_images:
                if actual_image.get_attribute('src') and 'http' in actual_image.get_attribute('src'):
                    image_urls.add(actual_image.get_attribute('src'))

            image_count = len(image_urls)

            if len(image_urls) >= max_links_to_fetch:
                print(f"Found: {len(image_urls)} image links, done!")
                break
        else:
            print("Found:", len(image_urls), "image links, looking for more ...")
            time.sleep(30)
            load_more_button = wd.find_element_by_css_selector(".mye4qd")
            if load_more_button:
                wd.execute_script("document.querySelector('.mye4qd').click();")

        # move the result startpoint further down
        results_start = len(thumbnail_results)

    return list(image_urls)


class HoldoutFetcher(object):

    def __init__(self, cloud_classes, holdout_path):

        self.query_list = cloud_classes
        self.holdout_path = holdout_path

        chromedriver_path = CHROMEDRIVER_PATH

        self.driver = webdriver.Chrome(
            executable_path=chromedriver_path
        )
        self.metadata_df = None

    def generate_metadata(self, n_images):

        if not os.path.isdir(self.holdout_path):
            os.mkdir(self.holdout_path)

        metadata_df = self._build_holdout_metadata(n_images)
        metadata_df.to_csv(os.path.join(self.holdout_path, "holdout_metadata.csv"))
        self.metadata_df = metadata_df

    def download_images(self):

        if not isinstance(self.metadata_df, pd.DataFrame):
            assert "Need to run generate_metadata() before downloading"

        self._download_from_holdout_meta()

    def _build_holdout_metadata(self,
                                n_images=10):

        holdout_metadata = []
        for q in self.query_list:
            # query to send to google image search
            images_query = q + " cloud photograph"

            image_urls = fetch_image_urls(
                query=images_query,
                max_links_to_fetch=n_images,
                sleep_between_interactions=1,
                wd=self.driver
            )

            image_ids = [str(uuid.uuid4()) for e in image_urls]
            image_names = [q] * len(image_ids)
            q_df = pd.DataFrame({
                "tag_class": image_names,
                "id": image_ids,
                "url": image_urls}
            )
            holdout_metadata.append(q_df)

        return pd.concat(holdout_metadata)

    def _download_from_holdout_meta(self):

        if not os.path.isdir(self.holdout_path):
            os.mkdir(self.holdout_path)

        for i, row in self.metadata_df.iterrows():

            image_url = row["url"]
            image_id = row["id"]
            image_tag = row["tag_class"]

            destination_folder = os.path.join(self.holdout_path, image_tag)
            if not os.path.isdir(destination_folder):
                os.mkdir(destination_folder)
            destination_filename = os.path.join(
                destination_folder, "{}_image.jpg".format(image_id)
            )
            download_image_from_url(image_url, destination_filename)
