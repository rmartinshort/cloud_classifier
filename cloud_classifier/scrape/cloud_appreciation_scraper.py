import requests
from selenium import webdriver
from bs4 import BeautifulSoup
import re

from cloud_classifier.cloud_classifier.common.constants import (
    GALLERY_URL,
    CHROMEDRIVER_PATH
)


class ExtractFromCloudAppreciationSite(object):

    def __init__(self):

        self.gallery_url = GALLERY_URL
        chromedriver_path = CHROMEDRIVER_PATH

        self.driver = webdriver.Chrome(
            executable_path=chromedriver_path
        )

        self.driver.get(self.gallery_url)

        self.advance_button = self.driver.find_element_by_xpath(
            "/html/body/div[2]/div/article/div/div[2]/div[2]/div/a"
        )
        self.image_regex = re.compile(
            'item post3 image*'
        )

    def get_driver(self):
        """

        :return:
        """

        return self.driver

    def advance_gallery(self):
        """

        :return:
        """

        # Click the advance button to get more images
        self.driver.execute_script(
            "arguments[0].click();",
            self.advance_button
        )

    def get_souped_page(self):
        """

        :param driver:
        :return:
        """

        return BeautifulSoup(self.driver.page_source, 'html.parser')

    def extract_from_souped_page(self, souped_page):
        """

        :param souped_page:
        :return:
        """

        extracted_data = []

        for image_data in souped_page.find_all(
            "div",
            {"class": self.image_regex}
        ):

            image_data_str = str(image_data)

            image_path = re.findall(r'src=(.*?)style',
                                    image_data_str
                                    )
            if len(image_path) > 0:
                image_title = re.findall(r'data-title=(.*?)\xa0',
                                         image_data_str
                                         )
                spans = image_data.find_all(name="div",
                                            class_="slideshowtags"
                                            )
                caption_tags = self._extract_caption_tag(spans)
                image_dict = {
                    "url": image_path,
                    "image_title": image_title,
                    "image_tags": caption_tags
                }
                extracted_data.append(image_dict)

        return extracted_data

    @staticmethod
    def _extract_caption_tag(text):
        """

        :param text:
        :return:
        """

        if not isinstance(text, str):
            text = str(text)

        tags = [x.split(">")[1] for x in re.findall(r'rel=(.*?)</a>', text)]
        return tags
