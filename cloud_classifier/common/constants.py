import os

GALLERY_URL = "https://cloudappreciationsociety.org/gallery/"
CHROMEDRIVER_PATH = "/Users/rmartinshort/Documents/DS_projects/cloud_classifier/chromedriver/chromedriver"
IMAGE_PATH = "/Users/rmartinshort/Documents/DS_projects/cloud_classifier/cloud_classifier/cloud_classifier/cloud_classifier/dataset"
METADATA_PATH = os.path.join(IMAGE_PATH, "metadata.csv")
HOLDOUT_PATH = os.path.join(IMAGE_PATH, "holdout")

scrape_tags_to_remove = [
    "clouds that look like things",
    "a classic example",
    "our favourites",
    "app photo of the day"
]
