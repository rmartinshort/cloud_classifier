from cloud_classifier.cloud_classifier.scrape.cloud_appreciation_scraper import (
    ExtractFromCloudAppreciationSite
)
from cloud_classifier.cloud_classifier.comon.constants import METADATA_PATH
import time
import uuid
import pandas as pd


def clean_tag(tag):
    """

    :param tag:
    :return:
    """
    tag = tag.lower()
    tag = tag.replace("&amp;", "and")
    tag = tag.replace(" / ", " ")
    return tag

def clean_image_tags(tags_list):
    """

    :param tags_list:
    :return:
    """
    return [clean_tag(t) for t in tags_list]

def generate_images_database(image_dict):
    """

    :param image_dict:
    :return:
    """
    images_df = {
        "id": [],
        "url": [],
        "title": [],
        "tags": []
    }

    for _image in image_dict:
        unique_id = str(uuid.uuid4())

        try:
            image_url = _image["url"][0].replace('"', "")
            image_title = _image["image_title"][0].replace('"', "").replace(",", ";")
            image_tags = clean_image_tags(_image["image_tags"])

            if isinstance(image_url, str):
                image_url = image_url.split()[0]

            images_df["id"].append(unique_id)
            images_df["url"].append(image_url)
            images_df["title"].append(image_title)
            images_df["tags"].append(image_tags)
        except Exception as e:
            continue

    images_df = pd.DataFrame(images_df)

    return images_df

def scaper_main(iterations=5):
    """

    :param iterations:
    :return:
    """

    scraper = ExtractFromCloudAppreciationSite()

    for i in range(iterations):
        # wait 10 seconds for the page to load, then advance the gallery
        # at the end of the advancement process, scrape the entire page
        time.sleep(10)
        print("Iteration {}: Advancing gallery".format(i))
        scraper.advance_gallery()

    gallery_page = scraper.get_souped_page()
    image_data = scraper.extract_from_souped_page(gallery_page)

    return image_data

def saver_main(iterations=5):
    """
    Use as follows:
    >from cloud_classifier.cloud_classifier.scrape.scrape_driver import saver_main
    >saver_main(iterations=20)
    :param iterations:
    :return:
    """

    scraped_data = scaper_main(iterations=iterations)
    images_df = generate_images_database(scraped_data)
    images_df.to_csv(METADATA_PATH,index=False)
