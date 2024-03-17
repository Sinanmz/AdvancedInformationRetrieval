from requests import get
from bs4 import BeautifulSoup
from collections import deque
from concurrent.futures import ThreadPoolExecutor, wait
from threading import Lock
import json

from time import sleep


class IMDbCrawler:
    """
    put your own user agent in the headers
    """
    headers = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36',
        'Accept-Language': 'en-US'
    }
    top_250_URL = 'https://www.imdb.com/chart/top/'

    def __init__(self, crawling_threshold=1000):
        """
        Initialize the crawler

        Parameters
        ----------
        crawling_threshold: int
            The number of pages to crawl
        """
        # TODO
        self.crawling_threshold = crawling_threshold
        self.not_crawled = deque()
        self.crawled = []
        self.added_ids = deque()
        self.add_list_lock = Lock()
        self.add_queue_lock = Lock()
        self.pop_lock = Lock()

    def get_id_from_URL(self, URL):
        """
        Get the id from the URL of the site. The id is what comes exactly after title.
        for example the id for the movie https://www.imdb.com/title/tt0111161/?ref_=chttp_t_1 is tt0111161.

        Parameters
        ----------
        URL: str
            The URL of the site
        Returns
        ----------
        str
            The id of the site
        """
        # TODO
        # return URL.split('/')[4]
        parts = URL.split('/')
        for part in parts:
            if part.startswith('tt'):
                return part
        raise ValueError("IMDB ID Not Found")

    def write_to_file_as_json(self):
        """
        Save the crawled files into json
        """
        # TODO
        # pass
        crawled_path = "data/IMDB_Crawled.json"
        with open(crawled_path, 'w') as file:
            json.dump(list(self.crawled), file, indent=4)
        

        not_carawled_path = "data/IMDB_Not_Crawled.json"
        data = {
            'added_ids': list(self.added_ids),
            'not_crawled': list(self.not_crawled)
        }
        with open(not_carawled_path, 'w') as file:
            json.dump(data, file, indent=4)
        
        
    def read_from_file_as_json(self):
        """
        Read the crawled files from json
        """
        # TODO
        try:
            crawled = []
            not_crawled = []
            added_ids = []
            with open('data/IMDB_crawled.json', 'r') as f:
                crawled = list(json.load(f))

            with open("data/IMDB_Not_Crawled.json", 'r') as f:
                data = json.load(f)
                added_ids = deque(data.get('added_ids', []))
                not_crawled = deque(data.get('not_crawled', []))
            
            if len(crawled) > 0 and len(not_crawled) > 0 and len(added_ids) > 0:
                self.crawled = crawled
                self.added_ids = added_ids
                self.not_crawled = not_crawled
        except:
            pass

    def crawl(self, URL):
        """
        Make a get request to the URL and return the response

        Parameters
        ----------
        URL: str
            The URL of the site
        Returns
        ----------
        requests.models.Response
            The response of the get request
        """
        # TODO
        temp = ['second', 'third', 'fourth', 'fifth', 'sixth', 'seventh', 'eighth', 'ninth', 'tenth']
        j = 0
        for j in range(10):
            try:
                for i in range(10):
                    response = get(URL, headers=self.headers)
                    if response.status_code == 200:
                        return response
                    else:
                        if i < 9:
                            print(f"Get request to {URL}, returned this unexpected status code: {response.status_code}")
                            print(f"Trying for the {temp[i]} time ...")
                            sleep(2)

                print(f"Get request to {URL} failed after trying for ten times.")    
                return None
            except Exception as e:
                if j < 9:
                    print(f"An error occured crawling {URL} with exception: {e}")
                    print(f"Trying for the {temp[j]} time ...")
                    sleep(2)
        print(f"Get request to {URL} failed after trying for ten times.")
        return None

    def extract_top_250(self):
        """
        Extract the top 250 movies from the top 250 page and use them as seed for the crawler to start crawling.
        """
        # TODO update self.not_crawled and self.added_ids
        response = self.crawl(self.top_250_URL)
        if response and response.status_code == 200:
            soup = BeautifulSoup(response.content, 'html.parser')
            movie_links = soup.select('ul .ipc-title-link-wrapper')

            with self.add_queue_lock:
                for link in movie_links:
                    url = 'https://www.imdb.com/'
                    hrf = link.get('href')
                    parts = hrf.split('/')
                    while True:
                        part = parts.pop(0) + '/'
                        url += part
                        if part.startswith('tt'):
                            url = url[:-1]
                            break
                    if self.get_id_from_URL(url) not in self.added_ids:
                        self.not_crawled.append(url)
                        self.added_ids.append(self.get_id_from_URL(url))
                        
                        
    def get_imdb_instance(self):
        return {
            'id': None,  # str
            'title': None,  # str
            'first_page_summary': None,  # str
            'release_year': None,  # str
            'mpaa': None,  # str
            'budget': None,  # str
            'gross_worldwide': None,  # str
            'rating': None,  # str
            'directors': None,  # List[str]
            'writers': None,  # List[str]
            'stars': None,  # List[str]
            'related_links': None,  # List[str]
            'genres': None,  # List[str]
            'languages': None,  # List[str]
            'countries_of_origin': None,  # List[str]
            'summaries': None,  # List[str]
            'synopsis': None,  # List[str]
            'reviews': None,  # List[List[str]]
        }

    def start_crawling(self):
        """
        Start crawling the movies until the crawling threshold is reached.
        TODO: 
            replace WHILE_LOOP_CONSTRAINTS with the proper constraints for the while loop.
            replace NEW_URL with the new URL to crawl.
            replace THERE_IS_NOTHING_TO_CRAWL with the condition to check if there is nothing to crawl.
            delete help variables.

        ThreadPoolExecutor is used to make the crawler faster by using multiple threads to crawl the pages.
        You are free to use it or not. If used, not to forget safe access to the shared resources.
        """

        # help variables
        # WHILE_LOOP_CONSTRAINTS = None
        # NEW_URL = None
        # THERE_IS_NOTHING_TO_CRAWL = None

        if len(self.crawled) == 0:
            self.extract_top_250()
        futures = []
        crawled_counter = 0

        with ThreadPoolExecutor(max_workers=8) as executor:
            while crawled_counter < self.crawling_threshold:
                self.pop_lock.acquire()
                URL = self.not_crawled.popleft()
                self.pop_lock.release()
                futures.append(executor.submit(self.crawl_page_info, URL))
                crawled_counter += 1
                if len(self.not_crawled) == 0:
                    wait(futures)
                    futures = []


    def crawl_page_info(self, URL):
        """
        Main Logic of the crawler. It crawls the page and extracts the information of the movie.
        Use related links of a movie to crawl more movies.
        
        Parameters
        ----------
        URL: str
            The URL of the site
        """
        print("new iteration")
        # TODO
        # pass
        response = self.crawl(URL)
        movie = self.get_imdb_instance()
        self.extract_movie_info(response, movie, URL)
        self.add_list_lock.acquire()
        try:
            self.crawled.append(movie)
        finally:
            self.add_list_lock.release()
        
        self.add_queue_lock.acquire()
        for link in movie['related_links']:
            if self.get_id_from_URL(link) not in self.added_ids:
                    self.not_crawled.append(link)
                    self.added_ids.append(self.get_id_from_URL(link))
        self.add_queue_lock.release()

        


    def extract_movie_info(self, res, movie, URL):
        """
        Extract the information of the movie from the response and save it in the movie instance.

        Parameters
        ----------
        res: requests.models.Response
            The response of the get request
        movie: dict
            The instance of the movie
        URL: str
            The URL of the site
        """
        # TODO

        main_soup = BeautifulSoup(res.content, 'html.parser')
        summary_res = self.crawl(self.get_summary_link(URL))
        summary_soup = BeautifulSoup(summary_res.content, 'html.parser')

        review_res = self.crawl(self.get_review_link(URL))
        review_soup = BeautifulSoup(review_res.content, 'html.parser')

        movie['id'] = self.get_id_from_URL(URL)
        movie['title'] = self.get_title(main_soup)
        movie['first_page_summary'] = self.get_first_page_summary(main_soup)
        movie['release_year'] = self.get_release_year(main_soup)
        movie['mpaa'] = self.get_mpaa(main_soup)
        movie['budget'] = self.get_budget(main_soup)
        movie['gross_worldwide'] = self.get_gross_worldwide(main_soup)
        movie['rating'] = self.get_rating(main_soup)
        movie['directors'] = self.get_director(main_soup)
        movie['writers'] = self.get_writers(main_soup)
        movie['stars'] = self.get_stars(main_soup)
        movie['related_links'] = self.get_related_links(main_soup)
        movie['genres'] = self.get_genres(main_soup)
        movie['languages'] = self.get_languages(main_soup)
        movie['countries_of_origin'] = self.get_countries_of_origin(main_soup)
        movie['summaries'] = self.get_summary(summary_soup)
        movie['synopsis'] = self.get_synopsis(summary_soup)
        movie['reviews'] = self.get_reviews_with_scores(review_soup)

    def get_summary_link(self, url):
        """
        Get the link to the summary page of the movie
        Example:
        https://www.imdb.com/title/tt0111161/ is the page
        https://www.imdb.com/title/tt0111161/plotsummary is the summary page

        Parameters
        ----------
        url: str
            The URL of the site
        Returns
        ----------
        str
            The URL of the summary page
        """
        try:
            # TODO
            # pass
            if url.endswith('/'):
                return url + 'plotsummary'
            else:
                return url + '/plotsummary'
        except:
            print("failed to get summary link")

    def get_review_link(self, url):
        """
        Get the link to the review page of the movie
        Example:
        https://www.imdb.com/title/tt0111161/ is the page
        https://www.imdb.com/title/tt0111161/reviews is the review page
        """
        try:
            # TODO
            # pass
            if url.endswith('/'):
                return url + 'reviews'
            else:
                return url + '/reviews'
        except:
            print("failed to get review link")

    def get_title(self, soup):
        """
        Get the title of the movie from the soup

        Parameters
        ----------
        soup: BeautifulSoup
            The soup of the page
        Returns
        ----------
        str
            The title of the movie

        """
        try:
            # TODO
            # pass
            title_element = soup.find('h1', {'data-testid':"hero__pageTitle"})
            title = title_element.text
            return title
        except:
            print("failed to get title")
            return 'N/A'

    def get_first_page_summary(self, soup):
        """
        Get the first page summary of the movie from the soup

        Parameters
        ----------
        soup: BeautifulSoup
            The soup of the page
        Returns
        ----------
        str
            The first page summary of the movie
        """
        try:
            # TODO
            # pass
            storyline_element = soup.find('p', {'data-testid':"plot"})
            storyline = storyline_element.text
            return storyline
        except:
            print("failed to get first page summary")
            return 'N/A'

    def get_director(self, soup):
        """
        Get the directors of the movie from the soup

        Parameters
        ----------
        soup: BeautifulSoup
            The soup of the page
        Returns
        ----------
        List[str]
            The directors of the movie
        """
        try:
            # TODO
            # pass

            cast_element = soup.find_all('li', {'data-testid':"title-pc-principal-credit"})
            directors = []
            for element in cast_element:
                try:
                    if element.find('span').text == 'Director' or element.find('span').text == 'Directors':
                        for dir in element.select('li'):
                            if dir.text not in directors:
                                directors.append(dir.text)
                except:
                    try:
                        if element.find('a').text == 'Director' or element.find('a').text == 'Directors':
                            for dir in element.select('li'):
                                if dir.text not in directors:
                                    directors.append(dir.text)
                    except:
                        pass
            return directors
        except:
            print("failed to get director")
            return []

    def get_stars(self, soup):
        """
        Get the stars of the movie from the soup

        Parameters
        ----------
        soup: BeautifulSoup
            The soup of the page
        Returns
        ----------
        List[str]
            The stars of the movie
        """
        try:
            # TODO
            # pass
            cast_element = soup.find_all('li', {'data-testid':"title-pc-principal-credit"})
            stars = []
            for element in cast_element:
                try:
                    if element.find('span').text == 'Star' or element.find('span').text == 'Stars':
                        for s in element.select('li'):
                            if s.text not in stars:
                                stars.append(s.text)
                except:
                    try:
                        if element.find('a').text == 'Star' or element.find('a').text == 'Stars':
                            for s in element.select('li'):
                                if s.text not in stars:
                                    stars.append(s.text)
                    except:
                        pass
            return stars
        except:
            print("failed to get stars")
            return []

    def get_writers(self, soup):
        """
        Get the writers of the movie from the soup

        Parameters
        ----------
        soup: BeautifulSoup
            The soup of the page
        Returns
        ----------
        List[str]
            The writers of the movie
        """
        try:
            # TODO
            # pass
            cast_element = soup.find_all('li', {'data-testid':"title-pc-principal-credit"})
            writers = []
            for element in cast_element:
                try:
                    if element.find('span').text == 'Writer' or element.find('span').text == 'Writers':
                        for w in element.select('li'):
                            if w.text not in writers:
                                writers.append(w.text)
                except:
                    try:
                        if element.find('a').text == 'Writer' or element.find('a').text == 'Writers':
                            for w in element.select('li'):
                                if w.text not in writers:
                                    writers.append(w.text)
                    except:
                        pass
            return writers

        except:
            print("failed to get writers")
            return []

    def get_related_links(self, soup):
        """
        Get the related links of the movie from the More like this section of the page from the soup

        Parameters
        ----------
        soup: BeautifulSoup
            The soup of the page
        Returns
        ----------
        List[str]
            The related links of the movie
        """
        try:
            # TODO
            # pass
            similar_element = soup.find('section', {'data-testid':"MoreLikeThis"})
            similar_element = similar_element.find('div', {'data-testid':"shoveler-items-container"})
            links = similar_element.select('a')
            urls = []
            for link in links:
                url = 'https://www.imdb.com/'
                href = link.get('href')
                parts = href.split('/')
                while True:
                    part = parts.pop(0) + '/'
                    url += part
                    if part.startswith('tt'):
                        url = url[:-1]
                        break
                urls.append(url)
            return urls
        except:
            print("failed to get related links")
            return []

    def get_summary(self, soup):
        """
        Get the summary of the movie from the soup

        Parameters
        ----------
        soup: BeautifulSoup
            The soup of the page
        Returns
        ----------
        List[str]
            The summary of the movie
        """
        try:
            # TODO
            # pass
            summaries_element = soup.find('div', {'data-testid':"sub-section-summaries"})
            summaries_elements = summaries_element.find_all('div', {'class': 'ipc-html-content-inner-div'})
            summaries = []
            for summary in summaries_elements:
                summaries.append(summary.text)

            return summaries
        except:
            print("failed to get summary")
            return []

    def get_synopsis(self, soup):
        """
        Get the synopsis of the movie from the soup

        Parameters
        ----------
        soup: BeautifulSoup
            The soup of the page
        Returns
        ----------
        List[str]
            The synopsis of the movie
        """
        try:
            # TODO
            # pass
            synopsises_element = soup.find('div', {'data-testid':"sub-section-synopsis"})
            synopsises_elements = synopsises_element.find_all('div', {'class': 'ipc-html-content-inner-div'})
            synopsises = []
            for synopsis in synopsises_elements:
                synopsises.append(synopsis.text)

            return synopsises
        except:
            print("failed to get synopsis")
            return []

    def get_reviews_with_scores(self, soup):
        """
        Get the reviews of the movie from the soup
        reviews structure: [[review,score]]

        Parameters
        ----------
        soup: BeautifulSoup
            The soup of the page
        Returns
        ----------
        List[List[str]]
            The reviews of the movie
        """
        try:
            # TODO
            # pass
            reviews = []

            all = soup.find('div', {'class': 'lister-list'})
            elements = all.find_all('div', {'class': 'lister-item-content'})
            for element in elements:
                try:
                    rating = element.find('span', {'class': 'rating-other-user-rating'}).text.replace('\n', '')
                    review = element.find('div', {'class': 'text show-more__control'}).text
                    reviews.append([review, rating])
                except: 
                    continue

            return reviews
        except:
            print("failed to get reviews")
            return []

    def get_genres(self, soup):
        """
        Get the genres of the movie from the soup

        Parameters
        ----------
        soup: BeautifulSoup
            The soup of the page
        Returns
        ----------
        List[str]
            The genres of the movie
        """
        try:
            # TODO
            # pass

            genre_element = soup.find('div', {'data-testid':"genres"})
            genres = []
            for genre in genre_element.select('span'):
                genres.append(genre.text)
            return genres
        except:
            print("Failed to get generes")
            return []

    def get_rating(self, soup):
        """
        Get the rating of the movie from the soup

        Parameters
        ----------
        soup: BeautifulSoup
            The soup of the page
        Returns
        ----------
        str
            The rating of the movie
        """
        try:
            # TODO
            # pass
            reting_element = soup.find('div', {'data-testid':"hero-rating-bar__aggregate-rating__score"})

            rating = reting_element.text
            return rating
        except:
            print("failed to get rating")
            return 'N/A'

    def get_mpaa(self, soup):
        """
        Get the MPAA of the movie from the soup

        Parameters
        ----------
        soup: BeautifulSoup
            The soup of the page
        Returns
        ----------
        str
            The MPAA of the movie
        """
        try:
            # TODO
            # pass
            data = soup.find('div', {'class':"sc-491663c0-3 bdjVSf"})
            group = data.find('ul', {'class': 'ipc-inline-list ipc-inline-list--show-dividers sc-d8941411-2 cdJsTz baseAlt'})
            mpaa = 'N/A'
            for element in group.select('li'):
                link = element.select_one('a')
                try:
                    href = link.get('href')
                    if 'parentalguide' in href:
                        mpaa = link.text
                except:
                    pass
            return mpaa
        except:
            print("failed to get mpaa")
            return 'N/A'

    def get_release_year(self, soup):
        """
        Get the release year of the movie from the soup

        Parameters
        ----------
        soup: BeautifulSoup
            The soup of the page
        Returns
        ----------
        str
            The release year of the movie
        """
        try:
            # TODO
            # pass

            data = soup.find('div', {'class':"sc-491663c0-3 bdjVSf"})
            group = data.find('ul', {'class': 'ipc-inline-list ipc-inline-list--show-dividers sc-d8941411-2 cdJsTz baseAlt'})
            release_year = 'N/A'
            for element in group.select('li'):
                link = element.select_one('a')
                try:
                    href = link.get('href')
                    if 'releaseinfo' in href:
                        release_year = link.text
                except:
                    pass
            return release_year
        
        except:
            print("failed to get release year")
            return 'N/A'

    def get_languages(self, soup):
        """
        Get the languages of the movie from the soup

        Parameters
        ----------
        soup: BeautifulSoup
            The soup of the page
        Returns
        ----------
        List[str]
            The languages of the movie
        """
        try:
            # TODO
            # pass
            languages_element = soup.find('li', {'data-testid':"title-details-languages"})
            langauges = []
            for lang in languages_element.select('li'):
                langauges.append(lang.text)
            return langauges
        except:
            print("failed to get languages")
            return []

    def get_countries_of_origin(self, soup):
        """
        Get the countries of origin of the movie from the soup

        Parameters
        ----------
        soup: BeautifulSoup
            The soup of the page
        Returns
        ----------
        List[str]
            The countries of origin of the movie
        """
        try:
            # TODO
            # pass
            countries_element = soup.find('li', {'data-testid':"title-details-origin"})
            countries = []
            for cont in countries_element.select('li'):
                countries.append(cont.text)
            return countries
        except:
            print("failed to get countries of origin")
            return []

    def get_budget(self, soup):
        """
        Get the budget of the movie from box office section of the soup

        Parameters
        ----------
        soup: BeautifulSoup
            The soup of the page
        Returns
        ----------
        str
            The budget of the movie
        """
        try:
            # TODO
            # pass
            budget_element = soup.find('li', {'data-testid':"title-boxoffice-budget"})
            budget = budget_element.find('div').text
            return budget
        except:
            print("failed to get budget")
            return 'N/A'

    def get_gross_worldwide(self, soup):
        """
        Get the gross worldwide of the movie from box office section of the soup

        Parameters
        ----------
        soup: BeautifulSoup
            The soup of the page
        Returns
        ----------
        str
            The gross worldwide of the movie
        """
        try:
            # TODO
            # pass
            gross_element = soup.find('li', {'data-testid':"title-boxoffice-cumulativeworldwidegross"})
            gross = gross_element.find('div').text
            return gross
        except:
            print("failed to get gross worldwide")
            return 'N/A'


def main():
    imdb_crawler = IMDbCrawler(crawling_threshold=500)
    imdb_crawler.read_from_file_as_json()
    imdb_crawler.start_crawling()
    imdb_crawler.write_to_file_as_json()


if __name__ == '__main__':
    main()
