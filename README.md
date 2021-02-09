# Steam-Recommender-System
Game purchase recommender system

The requirement is to develop a recommender system based upon a dataset of Steam user behaviours. Steam
is a popular PC Gaming hub, with over 6000 games and community of gamers. The particulars of the
dataset are as follows:
1. The subject dataset will be taken from (https://www.kaggle.com/tamber/steam-video-games).
2. This dataset lists steam users behaviours with the following columns:
• user-id
• game-title
• behaviour-name - Displays either 'Purchase' or 'Play'
• value - If Play data it contains the number of hours played. If Purchase data it contains the
number 1.0, indicating that the game was purchased by that player.

Recommender System Implementation
Two (2) recommender algorithms where implemented. One based on user purchases and another based
on the hours played by the user.

Purchases Recommender-
This algorithm only uses game purchases to recommend new games while disregarding the number of
hours played.
The logic of this algorithm is implemented in the purchase_recommend(purchase_list) function of the
python code. This function is called with a list of strings as an argument. This argument list
should contain the list of games that a user has already purchased and which we want to base our
recommendation upon. The algorithm works as follows:
1. Computes a similarity matrix/array using Jaccard Distance. This computes the similarity between the
argument list passed with the function call and all other user purchase lists in the loaded Dataframe
(steam_purchaseList.csv). (Item-based similarity).
2. The similarity matrix is sorted in order of decreasing similarity.
3. The 200 most similar purchase lists are selected from the similarity matrix. These 200 lists are
concatenated and will be the basis of recommendation.
4. The 20 most frequently occurring games from the concatenated purchase lists are selected as the
top 20 recommendations. This could be easily expanded to 30, 40, etc. or reduced according to user
needs.

Play Hours Recommender - 
This algorithm uses the number of hours played as the basis for making game recommendations.
The logic of this algorithm is implemented in the play_recommend(game_hours) function. This function is called with a list of tuples as an argument. Each tuple contains the
name of a game and the number of hours played. For example a typical function call would be as
follows:
# Recommend based on play hours of "Counter-Strike" and "DoA"
rec = play_recommend([("Counter-Strike", 100.0), ("DoA", 52.5)])
# Recommend based on play hours of "Counter-Strike" only
rec = play_recommend([("Counter-Strike", 100.0), ("DoA", 52.5)])

The algorithm proceeds as follows:
1. For each tuple in the argument list we do the following:
• Compute a similarity matrix/array (Euclidean, based on hours played). This computes the
similarity between the hours played of the current game and all other players of the game.
(User-based similarity).
• The similarity matrix is sorted in order of decreasing similarity.
• The 200 most similar user behaviours (hours played) are selected from the similarity matrix.
These 200 lists are concatenated and are used as the basis for recommendation.
2. The 20 most frequently occurring games from the concatenated recommendation list are selected
as the top 20 recommendations. This could easily be expanded to the 30, 40, etc. or reduced
according to user system needs.

