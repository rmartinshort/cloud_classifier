from cloud_classifier.cloud_classifier.scrape.cloud_appreciation_scraper import (
    ExtractFromCloudAppreciationSite
)
from cloud_classifier.cloud_classifier.common.constants import (
    IMAGE_PATH,
    HOLDOUT_PATH,
    METADATA_PATH
)
from cloud_classifier.cloud_classifier.model.dataloader import (
    load_metadata
)
from cloud_classifier.cloud_classifier.scrape.google_images_scraper import (
    HoldoutFetcher
)
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

def holdout_main(
    holdout_path=HOLDOUT_PATH,
    train_path=IMAGE_PATH,
    n_images=10):

    """
    Use as follows
    >from cloud_classifier.cloud_classifier.scrape.scrape_driver import holdout_main
    >holdout_main(n_images=20)
    :param holdout_path:
    :param train_path:
    :param n_images:
    :return:
    """

    try:
        _, cloud_classes = load_metadata(
            replicate=0,
            image_path=train_path
        )
    except Exception as e:
        print(e)
        print("Need to run saver_main() and generate to_download.csv first. "
              "This generates the dataset that will be used to train the model"
              )
    holdout_loader = HoldoutFetcher(
                cloud_classes,
                holdout_path=holdout_path
    )

    holdout_loader.generate_metadata(n_images=n_images)
    holdout_loader.download_images()




