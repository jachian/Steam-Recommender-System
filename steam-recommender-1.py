# -*- coding: utf-8 -*-
"""
Created on Mon Dec 16 05:59:38 2019

@author: HUB HUB
"""
#global imports
import numpy as np
import pandas as pd


#global variables
#load stats dataframe
stats_df = pd.read_csv("steam_stats.csv", delimiter = ",")
    
#load utility matrix_df
utility_df = pd.read_csv("steam_utility_m.csv", delimiter = ",")
    
#load the purchases datafram 
purchases_df = pd.read_csv("steam_purchaseList.csv", delimiter = ",")

dic = {'Ben There' : 'Ben There, Dan That!'}

# game_hours = []


#####################################################################################################################
#
# This function is used for loading csv documents into Pandas dataframes
#
# Arguments - file_name - The name of the file to load
#           - delimiter - the character delimiter in quites that we want to divide file by.
#
# Returns - data - The pandas dataframe with the content loaded
#
##################################################################################################################
def load_data(file_name, delimit):
    
    #read into a data-frame
    data = pd.read_csv(file_name, delimiter = delimit)
    
    #convert the datframe to anumpy array and return
    #return data.values
    return data


#####################################################################################################################
#
# Was used in data preprocessing to load generate utility matrix of the data and write it to a csv file.
#
# Arguments - file_name - The name of the file to load
#           - delimiter - the character delimiter in quites that we want to divide file by.
#
# Returns - utility_matrix - The pandas dataframe with the utility matrix.
#
##################################################################################################################
def pre_process(file_name, delimit):
    
    # load the data from the original csv file ###########################################
    #data = load_data("steam-200k.csv", ",")
    data = load_data(file_name, delimit)
    
    #retrieve the list of games - no duplicates
    games = list(set(data['The Elder Scrolls V Skyrim']))
    #sort the list
    games.sort()
    
    #retireve the list of unique player IDs
    user = list(set(data['151603712']))
    #sort the players by ID
    user.sort()
    
    #create numpy array for data frame columns
    table_head = ['User'] + games
    
    
    #create the utility matrix dataframe  ###################################################
    utility_matrix = pd.DataFrame(columns = table_head)
    
    
    # for each user we get generate its utility matrix row and insert it into the utility matrix.
    for i in range(0, len(user)):
        # we first get the items concerned
        records = data.loc[data['151603712'] == user[i]]
        
        print(" Records == ")
        print(str(records))
        
        row = get_utility_row(user[i], records, games)
        
        # print("row " +str(i)+ " == ")
        # print(str(row))
        
        utility_matrix.loc[i] = row
    
    
    # write the utility matrix to a csv file
    export_csv = utility_matrix.to_csv(r'C:\Users\HUB HUB\Google Drive\Data Science\Data Mining\Project\steam_utility_m.csv', index = None, header=True)
    
    
    # return the processed data/ Dataframe
    return utility_matrix



#####################################################################################################################
#
# This is a helper function to the pre_process() function above
#
# Arguments - user_id - the user id sought
#           - records - the data row we are concerned with
#           - games_list - The list o games we want as columns
#
# Returns - A new utility matrix row
#
##################################################################################################################
def get_utility_row(user_id, records, games_list):
    
    print(" Records in utility_row == ")
    print(str(records))
    
    #create a dictionary of games ###################################################
    indexes = [i for i in range(0, len(games_list))]
    
    look_up = dict(zip(games_list, indexes))
    
    
    # get user records
    #data_frame[data_frame['151603712'] == user_id]  # this here does not work. try a for loop to manually get the respective items.
    
    #create user game purchase and play records
    record = [-1 for i in range(0, len(games_list))]
    
    for i in range(0, len(records)):
         #get game index from dictionary
         item = records.iloc[i][1]
         
         print("game item == " +item)
         
         game_index = look_up[item]
         
         #add the info to the record/game position
         record[game_index] = records.iloc[i][3]
    
    # when done return the the new record/row concatenated user ID with record
    return [user_id] + record



#####################################################################################################################
#
#  A helper function used to get a list of row from a dataframe with the specified user id
#
# Arguments - user_id - The id of the user we want the record for
#           - dataframe - the pandas dataframe we want to get the user rows from
#
# Returns - records - a list of pandas rows of the specified user id
#
##################################################################################################################
def get_user_records(user_id, dataframe):
   
    records = []
    
    for i in range(0, len(dataframe)):
        
        if dataframe.loc[i][0] == user_id:
            records.append(dataframe.loc[i])
          
    return records


#####################################################################################################################
#
# This is another data pre-processing function used to generate the user purchase list and write it to csv file
#
# Arguments - file_name - None
#
# Returns - purchase_list - The pandas dataframe with the purchase list
#
##################################################################################################################
def generate_purchase_lists():
    
    data = load_data("steam_utility_m.csv", ",")
    
    purchase_list = pd.DataFrame(index = data['User'], columns = ["User", "Purchases"])
    
    for i in range(0, len(data)):
        #generate purchase row and append it the purchase list
        row = []
        row.append(data.iloc[i][0])
        
        li = generate_list(data.iloc[i], data.columns)
        
        #append our result to the row
        row.append(li)
        
        print(" purchase list for  == ")
        print(str(row))
        
        purchase_list.loc[i] = row
        
    export_csv = purchase_list.to_csv(r'C:\Users\HUB HUB\Google Drive\Data Science\Data Mining\Project\steam_purchaseList.csv', index = None, header=True)  
    
    return purchase_list


#####################################################################################################################
#
# Was used in data preprocessing to load generate the stats matrix of the data and write it to a csv file.
#
# Arguments - None
#
# Returns - stats - The pandas dataframe with the stats matrix.
#
##################################################################################################################
def generate_stats_matrix():
    data = load_data("steam_utility_m.csv", ",")
        
    # get the game names or column names
    cols = data.columns
        
    #create the new dataframe
    stats = pd.DataFrame(index = ['sum', 'max', 'min', 'mean', 'std' ], columns = cols[1:])
    

        
    #loop through the utility matrix and generate the stats for each game
    for j in range(0, stats.shape[1]):
        
        col = data[cols[j + 1]]
        #
        print("col == ")
        print(str(col))
            
        values = statistics(list(col)) 
        
        stats.iloc[0][j] = values[0]        #sum
        stats.iloc[1][j] = max(list(col))   #max
        
        if values[1] == 99999.0:            #min
            stats.iloc[2][j] = 0.0
        else:
            stats.iloc[2][j] = values[1]
        
        stats.iloc[3][j] = values[2]        #mean
        stats.iloc[4][j] = values[3]        #std
        
        print("generate stats == ")
        print(str(values))
        
    export_csv = stats.to_csv(r'C:\Users\HUB HUB\Google Drive\Data Science\Data Mining\Project\steam_stats.csv', index = None, header=True)  
        
    return stats



#####################################################################################################################
#
# Was used in data preprocessing. A helper function to the generate_stats_matrix() function. used to
# compute the sum, min, mean and stdev of list of doubles.
#
# Arguments - lyst - the list of doubles to use for computation
#
# Returns - vals - A list of computed results such that (vals[0] == sum, vals[1] ==  min, vals[2] == mean, vals[3] == std)
#
##################################################################################################################    
def statistics(lyst):
    
    # vals[0] == sum, vals[1] ==  min, vals[2] == mean, vals[3] == std
    vals = [0.0, 99999.0, 1.0, 0.0]
    
    # if lyst is empty
    if lyst == []:
        return vals
    
    #used to count valid entries
    count = 0.0
    
    # compute sum and min
    for x in lyst:
        if x > 1.0:
            count = count + 1
            
            #update sum
            vals[0] = vals[0] + x
            
            #update min
            if x < vals[1]:
                vals[1] = x
            
    # compute mean
    if count != 0.0:
       vals[2] = vals[0] / count
    else:
       vals[2] = vals[0]
    
    # compute std
    for x in lyst:
        if x > 1.0:
            vals[3] = vals[3] + ((x - vals[2])**2.0)
        
    vals[3] = (vals[3])**0.5
    
    return vals


#####################################################################################################################
#
# Was used in data preprocessing. A helper used in generating te purchase list. used to ggenerate a list of user
# purchases
#
# Arguments - user_row - the user dataframe row we want to get purchases from
#             columns_list - the list of game column names to choose from
#
# Returns - A string- A stringified purchase list of games
#
##################################################################################################################
def generate_list(user_row, columns_list):
    
    #purchase set
    purchase_set = []
    
    for i in range(1, len(user_row)):
        if user_row[i] > 0.00:
            purchase_set.append(columns_list[i])
    
    
    print(" appended list == ")
    print(str(purchase_set))   
    
    return str(purchase_set)


#####################################################################################################################
#
# Used to convert a strinigfied list into python list of strings
#
# Arguments - string - the string to convert to a list
#
# Returns - lyst - The list of strings
#
##################################################################################################################
def string_to_list(string):
    
    if not string:
        return []
    
    #remove the outer brackets
    stry = string.strip('][')
    
    stry = stry.replace('"', "")
    stry = stry.replace('[', "")
    stry = stry.replace(']', "")
    # stry = stry.replace(' ', "")
    
    #now convert to list and return
    stry = stry.split(',')
    
    #get rid of white space at the front of words
    lyst = []
    lyst.append(remove_back(remove_chars(stry[0])))
    
    for i in range(1, len(stry)):
        lyst.append(remove_back(remove_chars(stry[i])))
        
    return lyst


#####################################################################################################################
#
# Used to remove spaces and quotes from the front of a string
#
# Arguments - string - the string to update
#
# Returns - string - The updated string
#
##################################################################################################################
def remove_chars(string):
    if string == "":
        return string
    
    case = 0
    
    if string[0] == ' ':  #"space at front
        case = 1 
        
        if string[1] == "'": #space at front and qoten at 1
            case = 2
        
    
    if string[0] == "'": #quote at 0
        case = 3
    
    if case == 1:
        return string[1:]
        
    if case == 2:
        return string[2:]
    
    if case == 3:
        return string[1:]
    
    return string 


#####################################################################################################################
#
# Used to remove quotes from the back of a string
#
# Arguments - string - the string to update
#
# Returns - string - The updated string
#
##################################################################################################################
def remove_back(string):
    
    if string == "":
        return string
    
    end = len(string) - 1
    
    if string[end] == "'":
        return string[:end]
    
    return string

    


# recommender system core  #############################################################################################


#####################################################################################################################
#
# This is called in order to compute a purchase similarity set based on jaccard distance
#
# Arguments - purchase_list - the list of games purchased
#
# Returns - list - The similarity list
#
##################################################################################################################
def purchase_similarity_matrix(purchase_list):
    
    if (purchase_list == []):
       return []
   
    #convert the purchase list into a set
    purchase_set = set(purchase_list)
     
    #create blank similarity list
    similarity = []
    
    #we look through through the steam purchases dataframe and pull the ames for each user to compute similarity
    for i in range(0, len(purchases_df)):
        steam_user_set = set(string_to_list(purchases_df.iloc[i][1]))
        
        jaccard = jaccard_similarity(purchase_set, steam_user_set)
        
        #append result to similarity list and the current index as a tuple
        similarity.append((jaccard, i))
    
    return similarity



#####################################################################################################################
#
# This is called in order to compute a play similarity set based on euclidean distance
#
# Arguments - game_name - The name of the game
#           - hours_played - The number of hours thegame was played
#
# Returns - list - The similarity list
#
##################################################################################################################
def play_similarity_matrix(game_name, hours_played):
    
    # make sure we dont have any empty parameters before proceeding
    if((game_name == "") or (hours_played <= 0.0) or (utility_df.empty)):
        return []
    
    # get the game's column of hours played as a list
    game_hours = utility_df[game_name]
    
    #create blank similarity list
    similarity = [(5000, i) for i in range(0, len(game_hours))]
    
    # compute the similarity of the current game hours with other users and store in the list
    for i in range(0, len(game_hours)):
        
        #make sure that the user has bought the game
        if game_hours[i] < 0.0:
            similarity[i] = (5000, utility_df.iloc[i][0])
        else:
            distance = euclidean_simularity_1D(hours_played, game_hours[i])
            
            #update distance matrix
            similarity[i] = (distance, utility_df.iloc[i][0])
        
    
    return similarity

#####################################################################################################################
#
# This is called in to generate a purchases based recommendation list
#
# Arguments - purchase_list - the list of games purchased
#
# Returns - list - The list of top 20 recommendation
#
##################################################################################################################
def purchase_recommend(purchase_list):
    
    #load the purchases dataframe
    # purchases_df = load_data("steam_purchaseList.csv", ",")
    
    #compute the purchase similarity matrix
    sim_matrix = purchase_similarity_matrix(purchase_list)
    
    #sort the sim_matrix in order of decreasing similarity
    sim_matrix.sort(reverse = True)
    
    #select the 200 most similar purchases as a basisc for making a recommendation
    sim_list = sim_matrix[: 200]
    
    #get the list of potential recommendations
    pot_rec = pot_purchase_rec(sim_list)   # this is giving problems
    
    #sort the potential recommendation in decreasing order of priority
    pot_rec.sort(reverse = True)
    
    # generate a set of unique items
    pot_set = set(pot_rec)
    
    #find the set difference so that we dont reference our own purchases
    pot_set = pot_set - set(purchase_list)
    
    #re-convert the set to a list for use
    pot_set = list(pot_set)
    
    # set up a count array to keep track of instaces of games
    count = [0 for i in range(0, len(pot_set))]
    
    # we count how many times each game was encountered
    for i in range(0, len(pot_set)):
        g_count = count_instances(pot_set[i], pot_rec)
        count[i] = g_count
    
    #we generate the list of top 10 recommendations
    recommendations = get_recommendations(pot_set, count, 10)
    
    #return the top 20 recommendations
    return recommendations[:20]



#####################################################################################################################
#
# This is called in to generate a play hours based recommendation list
#
# Arguments - game_hours - A list of tupes. Each tupls as follows ("Game Name", Hours_played)
#                          eg - rec = play_recommend([("Counter-Strike", 100.0), ("DoA", 52.5)])  
#
# Returns - list - The list of top 20 recommendation
#
##################################################################################################################
def play_recommend(game_hours):
    
    #load the utilty matrix data frame
    # utility_df = load_data("steam_utility_m.csv", ",")
    
    
    # purchases_df = load_data("steam_purchaseList.csv", ",")
    
    #an empty list for recommendations
    recommendations = []
    
    
    #loop through the game_hours list and compute the similarity matrices of each game
    for i in range(0, len(game_hours)):
        sim_matrix = play_similarity_matrix(game_hours[i][0], game_hours[i][1])
        
        #sort the sim_matrix in ascending order
        sim_matrix.sort()
        
        #select the 1000 most similar users per hours played
        sim_list = sim_matrix[: 200]
        
        # get the top 10  hours played from each user in the sim list
    
        #get the list of potential recommendations - this will have to change
        pot_rec = pot_play_rec(sim_list)  # we have reached here
        
        # we append the new set of recommendation based on
        for string in pot_rec:
            recommendations.append(string)
    
    #sort the potential recommendations
    pot_rec.sort()
    
    # generate a set of unique items
    pot_set = set(pot_rec)
    
    #find the set difference so that we dont reference our own purchases
    pot_set = pot_set - set([game_hours[i][0] for i in range(0, len(game_hours))])
    
    #re-convert the set to a list for use
    pot_set = list(pot_set)
        
    pot_count = [0 for i in range(0, len(pot_set))]
    
    for i in range(0, len(pot_set)):
        pot_count[i] = count_instances(pot_set[i], pot_rec)
        
    #get the final set of recommendations
    final_rec = get_recommendations(pot_set, pot_count, 200)
    
    #sort in order of decreasing hours played
    final_rec.sort(reverse = True)
    
    # get get the lists without the hours payed
    rec = []
    original = []
    for i in range(0, len(final_rec)):
        rec.append(final_rec[i][1])
    
    for i in range(0, len(game_hours)):
        original.append(game_hours[i][1])
    
    # rec = list(set(rec) - set(original))
    
    return rec[:20]
        

    
#####################################################################################################################
#
# This is used to compute jaccard similarity of to sets
#
# Arguments - a_set - The first set we want to compare
#           - b_set - The set we want to compare a_set with
#
# Returns - jac - A double value of the computed similarity
#
##################################################################################################################
def jaccard_similarity(a_set, b_set):
    
    num_union = len(a_set | b_set)
    num_intersection = len(a_set & b_set)
    
    jac = 1 - ((num_union - num_intersection) / (num_union))
    
    return jac


#####################################################################################################################
#
# This is used to compute euclidean similarity of doublees
#
# Arguments - a_hours - The first number of hours
#           - b_hours - The number of hours we want to compare b_hours with
#
# Returns - - A double value of the computed similarity
#
##################################################################################################################
def euclidean_simularity_1D(a_hours, b_hours):
    
    return ((a_hours - b_hours) ** 2) ** 0.5



#####################################################################################################################
#
# This is used to generate a list of potential purchase recommendations
#
# Arguments - similarity_list - the list of tupleswith similarity with other games (similarity_number, Game_name)
#
# Returns - potentials - A list of potential recommdations
#
##################################################################################################################
def pot_purchase_rec(similarity_list):
    
    potentials = []
    #run through similarity list and extract the purchases from each user
    for i in range(0, len(similarity_list)):
        index = similarity_list[i][1]
        
        print("pot_purchase_rec *** sim index = " +str(index))
        
        #get the purchases string
        purchases = purchases_df.iloc[index][1] 
        print("pot_purchase_rec *** purchase string = ")
        print(str(purchases))
        
        #convert the string to a list
        purchases = string_to_list(purchases)
        
        #append pruchases from this users to the potentials list (flatten?)
        for string in purchases:
            potentials.append(string)
 
    return potentials



#####################################################################################################################
#
# This is used to generate a list of potential play recommendations
#
# Arguments - similarity_list - the list of tupleswith similarity with other games (similarity_number, Game_name)
#
# Returns - game_hours - A list of tuples of recommendations (hours_played, game_name)
#
##################################################################################################################
def pot_play_rec(similarity_list):
    
    # we want to go through the similarity list and find the games that are most
    # played by those users and return them
    
    rec = []
    
    for i in range(0, len(similarity_list)):
        # get that users' row from the purchase matrix
        row_df = purchases_df.loc[purchases_df['User'] == similarity_list[i][1]]
        
        
        print("sim matrix entry == ")
        print(similarity_list[i][1])
        
        #get the list of purchases
        purchases = []
        
        if row_df.empty:
           purchases = []
        else:
           purchases = string_to_list(row_df.iloc[0][1])
           
        #print("purchase list == ")
        #print(purchases)
        
        #get the utitlity matrix row
        utility_row = utility_df.loc[utility_df['User'] == similarity_list[i][1]]
        
        # print the results
        #print("utility row row == ")
        #print(str(utility_row))
        
        #generate the (game hours, gmae name) tuples
        game_hours = []
        
        #for g in purchases:
            #game_hours.append((utility_row[g], g)) #problem here
            
        for i in range(0, len(purchases)):
            if(purchases[i] in utility_row.columns):
               j = utility_row.columns.get_loc(purchases[i])
               tup = (utility_row.iloc[0][j], purchases[i])   # this might be a problem
            
               # print the results
               print(" j == " +str(j))
               print(" purchases == " +str(purchases[i]))
               print("tuple == ")
               print(str(tup))
            
               game_hours.append(tup)
                 
        
        #we sort the tuple to find the ones most played
        game_hours.sort(reverse = True)
        print("game hours == " +str(game_hours))
        
    
    return game_hours[:200]


def commas(string):
    if string == "":
        return False
    
    for i in range(0, len(string)):
        if(string[i] == ','):
            return True
    
    return False

        

def df_to_list(dataframe_row):
    
    if(dataframe_row.empty):
        return []
    
    tuple_list = []
    
    for j in range(0, dataframe_row.shape[1]):
        tuple_list.append((dataframe_row.iloc[0][j], dataframe_row.columns.values[j]))
        
    return tuple_list


    
# this works
def count_instances(game, recommendation_list):
    
    count = 0
    
    for g in recommendation_list:
        if(g == game):
            count = count + 1
        
    return count


#####################################################################################################################
#
# This is a helper function to the Play and Purchase recommendations function
#
# Arguments - recommendation_list - the list of game names
#             count_list - a list of integers repesenting the number of times a game was encountered. ie recommendation_list[i] was encountered count_list[i] times
#             num_rec - the number or recommendations we wish to output
#
# Returns - recommendations - A list of the top (num_rec) recommendations
#
##################################################################################################################
def get_recommendations(recommendation_list, count_list, num_rec):
    
    if(recommendation_list == []) or (count_list == []):
        return []
    
    #get the max count
    max_count = max(count_list)
    count = 0
    
    recommendations = []
    
    while max_count > 0:
        
        if count == num_rec:
            break
        
        for i in range(0, len(count_list)):
            if(count_list[i] == max_count):
                recommendations.append(recommendation_list[i])
                count = count + 1
            
        max_count = max_count - 1
    
    return recommendations

                
            
#  All code/functions below are used to generate Data analyses ###########################################################################
def display_stats():
    
    # stats_df = load_data("steam_stats.csv", ",")
    
    game = stats_df.loc[0].idxmax()
    
    print("## Most Hours Played == " +game+ "##############")
    print("  Details:")
    
    data = stats_df[game]
    
    print("     Total hours played == " +str(data[0]))
    print("     Max hour played by a player == " +str(data[1]))
    print("     Min hours played by a player == " +str(data[2]))
    print("     Mean/Average hours played == " +str(data[3]))
    print("     StdDev hours == " +str(data[4]))
    print("#####################################3############")
    

    game = stats_df.loc[1].idxmax()
    
    print("## Max hours played by a Player == " +game+ "##############")
    print("  Details:")
    
    data = stats_df[game]
    
    print("     Total hours played == " +str(data[0]))
    print("     Max hour played by a player == " +str(data[1]))
    print("     Min hours played by a player == " +str(data[2]))
    print("     Mean/Average hours played == " +str(data[3]))
    print("     StdDev hours == " +str(data[4]))  
    print("##################################################")
    
    game = stats_df.loc[2].idxmax()
    
    print("## Min hours played by a player == " +game+ "#############")
    print("  Details:")
    
    data = stats_df[game]
    
    print("     Total hours played == " +str(data[0]))
    print("     Max hour played by a player == " +str(data[1]))
    print("     Min hours played by a player == " +str(data[2]))
    print("     Mean/Average hours played == " +str(data[3]))
    print("     StdDev hours == " +str(data[4]))
    print("###################################################")
    
    game = stats_df.loc[3].idxmax()
    
    print("## Max Mean/Average hour played == " +game+ "###########")
    print("  Details:")
    
    data = stats_df[game]
    
    print("     Total hours played == " +str(data[0]))
    print("     Max hour played by a player == " +str(data[1]))
    print("     Min hours played by a player == " +str(data[2]))
    print("     Mean/Average hours played == " +str(data[3]))
    print("     StdDev hours == " +str(data[4]))
    print("##################################################")
    
    game = stats_df.loc[4].idxmax()
    
    print("## Max Spread (StdDev) == " +game+ "#############")
    print("  Details:")
    
    data = stats_df[game]
    
    print("     Total hours played == " +str(data[0]))
    print("     Max hour played by a player == " +str(data[1]))
    print("     Min hours played by a player == " +str(data[2]))
    print("     Mean/Average hours played == " +str(data[3]))
    print("     StdDev hours == " +str(data[4]))
    print("#################################################")
          
    print("######################################################################################")
    
    game = stats_df.loc[0].idxmin()
          
    print("## Least Hours Played == " +game+ "##############")
    print("  Details:")
    
    data = stats_df[game]
    
    print("     Total hours played == " +str(data[0]))
    print("     Max hour played by a player == " +str(data[1]))
    print("     Min hours played by a player == " +str(data[2]))
    print("     Mean/Average hours played == " +str(data[3]))
    print("     StdDev hours == " +str(data[4]))
    print("#####################################3############")
    

    game = stats_df.loc[1].idxmin()
    
    print("## Minimun hours played by a Player == " +game+ "##############")
    print("  Details:")
    
    data = stats_df[game]
    
    print("     Total hours played == " +str(data[0]))
    print("     Max hour played by a player == " +str(data[1]))
    print("     Min hours played by a player == " +str(data[2]))
    print("     Mean/Average hours played == " +str(data[3]))
    print("     StdDev hours == " +str(data[4]))  
    print("##################################################")
    
    game = stats_df.loc[2].idxmin()
    
    print("## Min hours played by a player == " +game+ "#############")
    print("  Details:")
    
    data = stats_df[game]
    
    print("     Total hours played == " +str(data[0]))
    print("     Max hour played by a player == " +str(data[1]))
    print("     Min hours played by a player == " +str(data[2]))
    print("     Mean/Average hours played == " +str(data[3]))
    print("     StdDev hours == " +str(data[4]))
    print("###################################################")
    
    game = stats_df.loc[3].idxmin()
    
    print("## Min Mean/Average hour played == " +game+ "###########")
    print("  Details:")
    
    data = stats_df[game]
    
    print("     Total hours played == " +str(data[0]))
    print("     Max hour played by a player == " +str(data[1]))
    print("     Min hours played by a player == " +str(data[2]))
    print("     Mean/Average hours played == " +str(data[3]))
    print("     StdDev hours == " +str(data[4]))
    print("##################################################")
    
    game = stats_df.loc[4].idxmin()
    
    print("## Min Spread (StdDev) == " +game+ "#############")
    print("  Details:")
    
    data = stats_df[game]
    
    print("     Total hours played == " +str(data[0]))
    print("     Max hour played by a player == " +str(data[1]))
    print("     Min hours played by a player == " +str(data[2]))
    print("     Mean/Average hours played == " +str(data[3]))
    print("     StdDev hours == " +str(data[4]))
    print("#################################################")


def analysis1():
    
    # we want to get the top 200 games by Total hours played and run some analyses on them
    
    #get the games tupples with the average number of hours played
    if(stats_df.empty):
       return []
    
    total_list = []
    max_list = []
    min_list = []
    mean_list = []
    stdev_list = []
    
    for j in range(0, stats_df.shape[1]):
        total_list.append((stats_df.iloc[0][j], stats_df.columns.values[j]))
        max_list.append((stats_df.iloc[1][j], stats_df.columns.values[j]))
        min_list.append((stats_df.iloc[2][j], stats_df.columns.values[j]))
        mean_list.append((stats_df.iloc[3][j], stats_df.columns.values[j]))
        stdev_list.append((stats_df.iloc[4][j], stats_df.columns.values[j]))
    
    #sort the games in descrending order
    total_list.sort(reverse = True)
    max_list.sort(reverse = True)
    min_list.sort(reverse = True)
    mean_list.sort(reverse = True)
    stdev_list.sort(reverse = True)
    
    #We want to make compare purchase and play recommendations of the same game via graph plot. ####
    
    # set up distance array - this will store the distance between the purchase and play recommendations
    total_distance = []
    max_distance = []
    min_distance = []
    mean_distance = []
    stdev_distance = []
                    
    x_data = [x for x in range(0, 20)]
    
    for i in range(0, 20):
        sim1 = jaccard_similarity(set(purchase_recommend([total_list[i][1]])), set(play_recommend([(total_list[i][1], mean_list[i][0])])))
        sim2 = jaccard_similarity(set(purchase_recommend([max_list[i][1]])), set(play_recommend([(max_list[i][1], mean_list[i][0])])))
        sim3 = jaccard_similarity(set(purchase_recommend([min_list[i][1]])), set(play_recommend([(min_list[i][1], mean_list[i][0])])))
        sim4 = jaccard_similarity(set(purchase_recommend([mean_list[i][1]])), set(play_recommend([(mean_list[i][1], mean_list[i][0])])))
        sim5 = jaccard_similarity(set(purchase_recommend([stdev_list[i][1]])), set(play_recommend([(stdev_list[i][1], mean_list[i][0])])))        
        
        
        total_distance.append(sim1)
        max_distance.append(sim2)
        min_distance.append(sim3)
        mean_distance.append(sim4)
        stdev_distance.append(sim5)
        
    
    #plot individual graphs
    # hours, g_names = zip(*total_list)
    plot_analysis(x_data, total_distance, 'Total Hours Played')   
    
    #hours, g_names = zip(*max_list)
    plot_analysis(x_data, max_distance, 'Max Hours Played')   
    
    #hours, g_names = zip(*min_list)
    plot_analysis(x_data, min_distance, 'Min Hours Played')
    
    # hours, g_names = zip(*mean_list)
    plot_analysis(x_data, mean_distance, 'Mean Hours Played')
    
    #hours, g_names = zip(*stdev_list)
    plot_analysis(x_data, stdev_distance, 'Widest Spread (stDev)')
    
    

    
    # we now try to plot our data on a graph
    import matplotlib.pyplot as plt
    
    plt.plot(x_data, total_distance, color = 'red', label='Total Hours Played')
    plt.plot(x_data, max_distance, color = 'green', label='Max Hours Played') 
    plt.plot(x_data, min_distance, color = 'blue', label='Min Hours Played') 
    plt.plot(x_data, mean_distance, color = 'orange', label='Mean Hours Played') 
    plt.plot(x_data, stdev_distance, color = 'black', label='Stdev Hours Played')
    
    plt.title('Comparison of Purchase and Play Recommendations') 
    plt.xlabel('Games') 
    plt.ylabel('Jaccard Similarity')
    plt.legend()
    
    plt.show()
    
    # return games_list[:200]


def analysis2():
    
    # we want to get the top 4 games by standard deviation and plot their recommendation in increasing hours played.
    
    #get the games tupples with the average number of hours played
    if(stats_df.empty):
       return []
    
    stdev_list = []
    
    for j in range(0, stats_df.shape[1]):
        stdev_list.append((stats_df.iloc[4][j], stats_df.columns.values[j]))
    
    #sort the games in descrending order
    stdev_list.sort(reverse = True)
    
    #We want to make compare purchase and play recommendations of the same game via graph plot. ####
    
    # set up distance array - this will store the distance between the purchase and play recommendations
    stdev_distance = []
    stdev_distance1 = []
    stdev_distance2 = []
    stdev_distance3 = []
                    
    x_hours = []
    x_hours1 = []
    x_hours2 = []
    x_hours3 = []
    
    y_rec = []
    y_rec1 = []
    y_rec2 = []
    y_rec3 = []
    
    # set hours from min hours to max hours in 20 iterations
    stats = stats_df[stdev_list[0][1]]
    stats1 = stats_df[stdev_list[1][1]]
    stats2 = stats_df[stdev_list[2][1]]
    stats3 = stats_df[stdev_list[3][1]]
    
    # we set up values for the x axes #####################################################
    increment = (stats[1] - stats[2]) / 20  # the increment
    start = stats[2]      #start at min
    for i in range(0, 20):
        x_hours.append(start)
        start = start + increment
    
    increment = (stats1[1] - stats1[2]) / 20  # the increment
    start = stats1[2]      #start at min
    for i in range(0, 20):
        x_hours1.append(start)
        start = start + increment
    
    increment = (stats2[1] - stats2[2]) / 20  # the increment
    start = stats2[2]      #start at min
    for i in range(0, 20):
        x_hours2.append(start)
        start = start + increment
    
    increment = (stats3[1] - stats3[2]) / 20  # the increment
    start = stats3[2]      #start at min
    for i in range(0, 20):
        x_hours3.append(start)
        start = start + increment
    
        
    #make the recommendations and compute the similarity with the originals
    rec = list(set(play_recommend([(stdev_list[0][1], stats[2])])) - set(stdev_list[0][1]))
    rec1 = list(set(play_recommend([(stdev_list[1][1], stats1[2])])) - set(stdev_list[1][1]))
    rec2 = list(set(play_recommend([(stdev_list[2][1], stats2[2])])) - set(stdev_list[2][1]))
    rec3 = list(set(play_recommend([(stdev_list[3][1], stats3[2])])) - set(stdev_list[3][1]))
    
    for i in range(0, 20):
        #compute the recommendations
        y = list(set(play_recommend([(stdev_list[0][1], x_hours[i])])) - set(stdev_list[0][1]))
        y1 = list(set(play_recommend([(stdev_list[1][1], x_hours1[i])])) - set(stdev_list[1][1]))
        y2 = list(set(play_recommend([(stdev_list[2][1], x_hours2[i])])) - set(stdev_list[2][1]))
        y3 = list(set(play_recommend([(stdev_list[3][1], x_hours3[i])])) - set(stdev_list[3][1]))
        
        #Store the recommendations
        y_rec.append(y)
        y_rec1.append(y1)
        y_rec2.append(y2)
        y_rec3.append(y3)
        
        #compute and store the similarity
        stdev_distance.append(jaccard_similarity(set(rec), set(y)))
        stdev_distance1.append(jaccard_similarity(set(rec1), set(y1)))
        stdev_distance2.append(jaccard_similarity(set(rec2), set(y2)))
        stdev_distance3.append(jaccard_similarity(set(rec3), set(y3)))
    
    
    #we now try to print table of the outputs
    pd.options.display.width=None
    # pd.set_option('display.max_columns', 500)

    data_table = pd.DataFrame(
        {
              "Hours" : x_hours,
              str(stdev_list[0][1]) : y_rec,
              "Sim/Dist" : stdev_distance,

        })
    
    print(data_table)
    
    export_csv = data_table.to_csv(r'C:\Users\HUB HUB\Google Drive\Data Science\Data Mining\Project\data_table.csv', index = None, header=True)
    
    
    data_table1 = pd.DataFrame(
        {
              "Hours" : x_hours1,
              str(stdev_list[1][1]) : y_rec,
              "Sim/Dist" : stdev_distance1,
        })
    
    print(data_table1)
    
    export_csv = data_table1.to_csv(r'C:\Users\HUB HUB\Google Drive\Data Science\Data Mining\Project\data_table1.csv', index = None, header=True)

    data_table2 = pd.DataFrame(
        {
              "Hours" : x_hours2,
              str(stdev_list[2][1]) : y_rec,
              "Sim/Dist" : stdev_distance2,
        })
    
    print(data_table2)
    
    export_csv = data_table2.to_csv(r'C:\Users\HUB HUB\Google Drive\Data Science\Data Mining\Project\data_table2.csv', index = None, header=True)
    
    data_table3 = pd.DataFrame(
        {
              "Hours" : x_hours3,
              str(stdev_list[3][1]) : y_rec3,
              "Sim/Dist" : stdev_distance3,
        })
    
    export_csv = data_table3.to_csv(r'C:\Users\HUB HUB\Google Drive\Data Science\Data Mining\Project\data_table3.csv', index = None, header=True)
    
    print(data_table3)
    
    
    #plot graphs
    plot_analysis1(x_hours, stdev_distance, stdev_list[0][1])
    plot_analysis1(x_hours1, stdev_distance1, stdev_list[1][1])
    plot_analysis1(x_hours2, stdev_distance2, stdev_list[2][1])
    plot_analysis1(x_hours3, stdev_distance3, stdev_list[3][1])
    
    
    #x_hours_a = [x_hours, x_hours1, x_hours2, x_hours3]
    #distance_a = [stdev_distance, stdev_distance1, stdev_distance2, stdev_distance3]
    #names_a = [stdev_list[0][1], stdev_list[1][1], stdev_list[2][1], stdev_list[3][1]]
    
    plot_analysis1_all(x_hours, x_hours1, x_hours2, x_hours3,
                       stdev_distance, stdev_distance1, stdev_distance2, stdev_distance3,
                       stdev_list[0][1], stdev_list[1][1], stdev_list[2][1], stdev_list[3][1])
    
    # we now try to plot our data on a graph
    #import matplotlib.pyplot as plt
    
    #fig, (ax, ax1, ax2, ax3) = plt.subplots(3)
    #fig.suptitle('Differences in recommendations as play hours change')
    #ax.plot(x_hours, stdev_distance)
    #ax.title(stdev_list[0][1])
    #ax.ylabel('Jaccard Sim')



def plot_analysis(game_list, distance, plot_title):
    
    import matplotlib.pyplot as plt
    
    plt.plot(game_list, distance, color = 'blue', label = plot_title)

    plt.title('Differences between purchase and play hours recommeendations') 
    plt.xlabel('Games') 
    plt.ylabel('Jaccard Sim')
    plt.legend()
    
    plt.show()


def plot_analysis1(x_hours, distance, game_name):
    
    import matplotlib.pyplot as plt
    
    plt.plot(x_hours, distance, color = 'blue', label = game_name)

    plt.title('Recommendation differences as play hours increase') 
    plt.xlabel('Play Hours') 
    plt.ylabel('Jaccard Sim')
    plt.legend()
    
    plt.show()


def plot_analysis1_all(x_hours, x_hours1, x_hours2, x_hours3,
                       distance, distance1, distance2, distance3, 
                       game_name, game_name1, game_name2, game_name3):
    
    # we now try to plot our data on a graph
    import matplotlib.pyplot as plt
    
    plt.plot(x_hours, distance, color = 'red', label = game_name)
    plt.plot(x_hours1, distance1, color = 'green', label = game_name1) 
    plt.plot(x_hours2, distance2, color = 'blue', label = game_name2) 
    plt.plot(x_hours3, distance3, color = 'orange', label = game_name3) 

    plt.title('Comparison of recommendation differences of different games') 
    plt.xlabel('Play Hours') 
    plt.ylabel('Jaccard Sim')
    plt.legend()
    
    plt.show()
    


def apriori_analysis():
    
    from mlxtend.frequent_patterns import apriori
    from mlxtend.frequent_patterns import association_rules
    
    #make a copy of our concerned dataframe
    copy_df = utility_df.copy()
    
    #prepare dataframe for apriori
    for i in range(0, copy_df.shape[0]):
        for j in range(1, copy_df.shape[1]):
            
            if copy_df.iloc[i][j] <= 0:
                copy_df.iloc[i][j] = 0.0
            else:
                copy_df.iloc[i][j] = 1.0
    
    #we now try to get the frequent item sets
    frequent_40 = apriori(copy_df, min_support = 0.40, use_colnames=True)
    
    print("frequent items == " +str(len(frequent_40)))
    print(str(frequent_40))
    
    
    #now we generate the rules with the support confidence and lift
    rules = association_rules(frequent_40, metric = "lift", min_threshold = 1)
    
    
    return rules


    
    



#"['Counter-Strike', 'Day of Defeat', 'Deathmatch Classic', 'Half-Life', 'Half-Life Blue Shift', 'Half-Life Opposing Force', 'Ricochet', 'Team Fortress Classic']"
            
    



#def generate_stats_matrix():
    #data = load_data("steam_utility_m.csv", ",")
        
    # get the game names or column names
    #cols = data.columns
        
    #create the new dataframe
    #stats = pd.DataFrame(index = ['sum', 'max', 'min', 'mean', 'std' ], columns = cols[1:])
        
    #loop through the utility matrix and generate the stats for each game
   # for x in cols[1:]:
        #col = data[x]
        #
        #print("col == ")
       # print(str(col))
            
        #values = statistics(list(col))            
            
        #stats['sum'][x] = values[0]
        #stats['max'][x] = max(list(col))
        #stats['min'][x] = values[1]
        #stats['mean'][x] = values[2]
        #stats['std'][x] = values[3]
        
       # print("generate stats == ")
        #print(str(values))
        
   # export_csv = purchase_list.to_csv(r'C:\Users\HUB HUB\Google Drive\Data Science\Data Mining\Project\steam_stats.csv', index = None, header=True)  
        
    #return stats  */
    
    
    
    


    
    
    
    
    
    

