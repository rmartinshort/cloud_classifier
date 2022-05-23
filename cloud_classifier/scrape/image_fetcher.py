import pandas as pd
import os
import requests # request img from web
import shutil # save img locally
from cloud_classifier.cloud_classifier.comon.constants import (
    METADATA_PATH,
    IMAGE_PATH,
    scrape_tags_to_remove
)


def download_image_from_url(url, file_name):
    """

    :param url:
    :param file_name:
    :return:
    """
    res = requests.get(url, stream=True)
    if res.status_code == 200:
        with open(file_name, 'wb') as f:
            shutil.copyfileobj(res.raw, f)
        print('Image sucessfully Downloaded: ', file_name)
    else:
        print('Image Couldn\'t be retrieved')

class ImageFetcher(object):

    def __init__(self):

        self.original_metadata_path = METADATA_PATH
        self.image_path = IMAGE_PATH
        self.metadata_df = self._load_metadata()
        self.to_download = None

    def assemble_images_to_download(self):

        redacted_metadata_df = self._load_metadata()
        labels_to_classify = self._redact_rare_tags()
        print(labels_to_classify)
        df_to_download = self._select_tags_to_classify(labels_to_classify)
        self.to_download = df_to_download

    def download_images(self):

        if isinstance(self.to_download,pd.DataFrame):
            self._download_images()

    def _load_metadata(self):

        df = pd.read_csv(
            self.original_metadata_path,
            converters={"tags": eval})
        df = df.explode("tags")
        # Remove the tags that aren't useful for classification
        df = df[~df["tags"].isin(scrape_tags_to_remove)]
        return df

    def _redact_rare_tags(self,nkeep=50):

        tags_to_classify = self.metadata_df.groupby(
            "tags"
        ).agg(
            {"id": "count"}
        ).reset_index(
        ).rename(
            columns={"id": "count"}
        ).sort_values(
            by="count"
        )
        tags_to_classify = tags_to_classify[tags_to_classify["count"] >= nkeep]
        labels_to_classify = list(tags_to_classify["tags"].unique())
        return labels_to_classify

    def _select_tags_to_classify(self,labels):

        df_tags = self.metadata_df.copy()
        df_tags["tags"] = self.metadata_df[
            "tags"
        ].apply(
            lambda x: "other" if x not in labels else x
        )

        tags_to_display = df_tags.groupby(
            "tags"
        ).agg(
            {"id": "count"}
        ).reset_index(
        ).rename(
            columns={"id": "count"}
        ).sort_values(
            by="count"
        )
        print(tags_to_display)

        final_df = df_tags.groupby(
            ["id", "url", "title"]
        )["tags"].apply(list).reset_index()

        return final_df


    def _download_images(self):

        for i, row in self.to_download.iterrows():
            unique_id = row["id"]
            url = row["url"]
            filename = "{}_image.jpg".format(unique_id)
            file_name = os.path.join(self.image_path, filename)
            download_image_from_url(url, file_name)


