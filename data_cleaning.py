"""
Columns to clean: Q2 number of ingrediants, Q4 cost, Q5 movies, Q6 drinks
"""
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from imdb import IMDb

stop_words = set(stopwords.words('english'))
ia = IMDb()

def detect_num_of_ingrediants(text: str) -> list[str]:
    numbers = re.findall(r"\d+", text)
    numbers = [int(num) for num in numbers]

    number_words = ["one", "two", "three", "four", "five", "six", "seven", 
                    "eight", "nine", "ten", "eleven", "twelve", "thirteen", 
                    "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", 
                    "nineteen", "twenty"]
    
    word_map = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, 
                    "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13, 
                    "fourteen": 14, "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18, 
                    "nineteen": 19, "twenty": 20}
    
    numbers += [word_map[word] for word in number_words if word in text.lower()]

    if numbers == []:
        ingredients_list = re.split(r",", text)
        return [len(ingredients_list)]

    return numbers

def detect_price(text: str) -> list[str]:
    numbers = re.findall(r"\d+(?:\.\d+)?", text)

    numbers = [float(num) for num in numbers if num != '']

    number_words = ["one", "two", "three", "four", "five", "six", "seven", 
                    "eight", "nine", "ten", "eleven", "twelve", "thirteen", 
                    "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", 
                    "nineteen", "twenty"]
    
    word_map = {"one": 1.0, "two": 2.0, "three": 3.0, "four": 4.0, "five": 5.0, "six": 6.0, "seven": 7.0, 
                    "eight": 8.0, "nine": 9.0, "ten": 10.0, "eleven": 11.0, "twelve": 12.0, "thirteen": 13.0, 
                    "fourteen": 14.0, "fifteen": 15.0, "sixteen": 16.0, "seventeen": 17.0, "eighteen": 18.0, 
                    "nineteen": 19.0, "twenty": 20.0, "quarter": 0.25, "half": 0.5, "dollar each": 1}
    
    numbers += [word_map[word] for word in number_words if word in text.lower()]

    return numbers

def detect_stopwords(text: str) -> bool:
    words = text.lower().split()
    return any(word in stop_words for word in words)

def detect_capitilized(text: str) -> list[str]:
    return re.findall(r'[A-Z][a-zA-Z]*', text)

def detect_movie(text: str) -> list[str]:
    common_movies = ['spirited away', 'american youg boy', 'stranger things', 'shawarma legend', 'finding nemo', 'jiro dreams of sushi', 'east side sushi', 'home alone',
                     'bbc food channel', 'ready player one', 'ratatouille', 'gilmore girls', 'goodfellas', 'chef', 'the avengers', 'middle eastern', 'mulan', 'za bebsi',
                     'isle of dogs', 'breakfast club', 'avengers', 'holes', 'shrek', 'hangover', 'polar express', 'elf', 'home alone', 'spider-man', 'superbad', 'monster inc',
                     'cloudy with a chance of meatballs', 'scott pilgrim vs the world', 'pleasant goat and big big wolf', 'minions: rise of gru', 'breaking bad', 'deadpool',
                     'wolverine', 'your name', 'penguins of madagascar']
    
    try:
        search = ia.search_movie(text.lower())
    except:
        print(f"Attempt {text} failed.")
        search = []
    
    if not search:
        search = [movie for movie in common_movies if movie in text.lower()]
    else:
        search = [movie['title'] for movie in search]

    return search

def detect_drinks(text: str) -> list[str]:
    common_drinks = ['7up', 'coca-cola', 'ice tea', 'iced tea', 'pepsi', 'fanta', 'mirinda', 'orange juice', 'ginger ale', 'lemonade', 'sprite', 
                     'kombucha', 'water', 'lemon water', 'pop', 'diet coke', 'ayran yogurt', 'beer', 'juice', 'wine', 'green tea', 'sake', 'tea', 'cocktail', 'bubble tea',
                     'crush', 'canada dry', 'red wine', 'coco cola', 'alcohol', 'chocolate milk', 'fanta', 'soft drink', 'sparkling water', 'mountain dew', 'no drink', 
                     'alcoholic drink', 'milk', 'coke', 'soda', 'cola', 'mango lassi', 'soy sauce']
    
    if detect_stopwords(text):
        return [drink for drink in common_drinks if drink in text.lower()]
    else:
        return [text]


def final_cleaning():
    df = pd.read_csv('cleaned_data_combined_modified.csv')
    df = df.rename(columns={"Q1: From a scale 1 to 5, how complex is it to make this food? (Where 1 is the most simple, and 5 is the most complex)": 'food_complexity', 
                            'Q2: How many ingredients would you expect this food item to contain?': 'num_ingredients', 
                            'Q3: In what setting would you expect this food to be served? Please check all that apply': 'serving_setting',
                            'Q4: How much would you expect to pay for one serving of this food item?': 'expected_cost', 
                            'Q5: What movie do you think of when thinking of this food item?': 'related_movie', 
                            'Q6: What drink would you pair with this food item?': 'paired_drink', 
                            "Q7: When you think about this food item, who does it remind you of?": 'associated_people',
                            'Q8: How much hot sauce would you add to this food item?': 'hot_sauce_level'})
    
    open("double_check.txt", "w", encoding="utf-8")

    for index, row in df.iterrows():
        output_file = []

        # Number of ingredients
        if isinstance(row["num_ingredients"], str):
            num_ingrediants = detect_num_of_ingrediants(row['num_ingredients'])
        else:
            num_ingrediants = []

        if len(num_ingrediants) > 0:
            df.at[index, 'num_ingredients'] = sum(num_ingrediants)/len(num_ingrediants)
        else:
            output_file.append(('num_ingredients', row['num_ingredients']))
            df.at[index, 'num_ingredients'] = None
        
        # Expected costs
        if isinstance(row["expected_cost"], str):
            prices = detect_price(row['expected_cost'])
        else:
            prices = []

        if len(prices) > 0:
            df.at[index, 'expected_cost'] = sum(prices)/len(prices)
        else:
            output_file.append(('expected_cost', row['expected_cost']))
            df.at[index, 'expected_cost'] = None
        
        # Detect drinks
        if isinstance(row["paired_drink"], str):
            drinks = detect_drinks(row['paired_drink'])
        else:
            drinks = []
        
        if len(drinks) > 0:
            df.at[index, 'paired_drink'] = drinks[0]
        else:
            output_file.append(('paired_drink', row['paired_drink']))
            df.at[index, 'paired_drink'] = None
        
        # Detect movies
        if isinstance(row["related_movie"], str):
            movies = detect_movie(row["related_movie"])
        
        if len(movies) > 0:
            df.at[index, 'related_movie'] = movies[0]
        else:
            output_file.append(('related_movie', row['related_movie']))
            df.at[index, 'related_movie'] = None

        # Write to file
        if output_file != []:
            with open("double_check.txt", "a", encoding="utf-8") as file:
                line = f"id: {index} "
                for name, comment in output_file:
                    line += f"{name}: {comment},"
                line += "\n"

                file.write(line)

    df.to_csv("clean_results.csv")

if __name__ == "__main__":
    #final_cleaning()
    df = pd.read_csv('clean_results.csv')
    df.hot_sauce_level = df.hot_sauce_level.fillna('None')
    print(df.isnull().sum())
    df.to_csv("clean_results.csv")