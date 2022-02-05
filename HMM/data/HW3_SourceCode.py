#!/usr/bin/env python
# coding: utf-8

# In[49]:


import pandas as pd
import csv
import json
from collections import ChainMap
from nltk.stem import WordNetLemmatizer
import random
import contractions
from nltk.tag import pos_tag


# In[50]:


df = pd.read_csv("train", sep='\t', quoting=csv.QUOTE_NONE, header=None)
df.loc[0] = [1, ".", "."]


# In[87]:


df.columns =['Index', 'Name', 'POS']


# In[52]:


from collections import Counter
results = Counter()
df['Name'].str.lower().str.split().apply(results.update)


# In[53]:


count_unk = 0
for word in list(results):
    if results[word]<2:
        key = pos_tag([word])[0][1]
        results["<unk_"+str(key).lower()+">"] += results[word]
        count_unk += results[word]
        del results[word]


# In[54]:


results = dict(results)


# In[55]:


sorted_results = {k: v for k, v in sorted(results.items(), key=lambda item: item[1], reverse = True)}
#print(sorted_results)


# In[89]:


unk_dict = [(key,value) for key, value in results.items() if '<unk_' in key.lower()]
count_unk = sum([int(pair[1]) for pair in unk_dict])
#print(count_unk)
#print(len(unk_dict))


# In[90]:


index=1
with open('vocab.txt', 'w') as vocab_file:
    #vocab_file.write("<unk>\t1\t"+str(sorted_results["<unk>"])+"\n")
    for temp in unk_dict:
        vocab_file.write(str(temp[0])+"\t"+str(index)+"\t"+str(temp[1])+"\n")
        index+=1


# In[91]:


index=14
with open('vocab.txt', 'a') as vocab_file:
    #vocab_file.write("<unk>\t1\t"+str(sorted_results["<unk>"])+"\n")
    for key,value in sorted_results.items():
        if '<unk_' in key.lower():
            continue
        else:
            vocab_file.write(key+"\t"+str(index)+"\t"+str(value)+"\n")
        index+=1


# In[84]:


vocab = set(sorted_results.keys())
#print(len(vocab))
#print(sorted_results['arizona'])


# In[57]:


def replaceWithUnk(word):
    if(word not in vocab):
        key = pos_tag([word])[0][1]
        return "<unk_"+key+">"
    else:
        return word


# In[58]:


df['Name'] = df['Name'].apply(lambda word : replaceWithUnk(word.lower()))
df['POS'] = df['POS'].apply(lambda word : word.lower())


# In[59]:


state_transiton_counts = {}
for i in range(1,len(df['POS'])):
    key = str(df['POS'][i-1])+"^"+str(df['POS'][i])
    if(key not in state_transiton_counts.keys()):
        state_transiton_counts[key] = 1
    else:
        state_transiton_counts[key]+=1


# In[60]:


state_counts = {}
for i in range(0,len(df['POS'])):
    key = str(df['POS'][i])
    if(key not in state_counts.keys()):
        state_counts[key] = 1
    else:
        state_counts[key]+=1


# In[61]:


#state_counts


# In[62]:


emission_transiton_counts = {}
for i in range(0,len(df['POS'])):
    key = str(df['POS'][i])+"^"+str(df['Name'][i])
    if(key not in emission_transiton_counts.keys()):
        emission_transiton_counts[key] = 1
    else:
        emission_transiton_counts[key]+=1


# In[63]:


#emission_transiton_counts


# In[64]:


transition_probabilites = {}
transition_matrix = ChainMap()
for key in state_transiton_counts.keys():
    backward,forward = key.split("^")
    probability = state_transiton_counts[key] / state_counts[backward]
    transition_key = "("+backward+","+forward+")"
    transition_probabilites[transition_key] = probability
    if backward not in transition_matrix.keys():
        #emisson_matrix[word] = [[state,probability]]
        transition_matrix[backward] = {forward:probability}
    else:
        transition_matrix[backward].update({forward:probability})
        #emisson_matrix[word].append([state,probability])


# In[92]:


#print(transition_probabilites)
len(transition_probabilites)


# In[66]:


#print(transition_matrix['.'])
#max(transition_matrix['cd'], key=transition_matrix['cd'].get)


# In[67]:


emission_probabilites = {}
emisson_matrix = ChainMap()
for key in emission_transiton_counts.keys():
    state,word = key.split("^")
    probability = emission_transiton_counts[key] / state_counts[state]
    transition_key = "("+state+","+word+")"
    emission_probabilites[transition_key] = probability
    if word not in emisson_matrix.keys():
        #emisson_matrix[word] = [[state,probability]]
        emisson_matrix[word] = {state:probability}
    else:
        emisson_matrix[word].update({state:probability})
        #emisson_matrix[word].append([state,probability])


# In[93]:


len(emission_probabilites)


# In[94]:


with open("hmm.json", "w") as outfile:
    #outfile.write("\nTransition Probabilites: \n")
    json.dump({"Transition Probabilites":transition_probabilites}, outfile)
    #outfile.write("\n\nEmission Probabilites: \n")
    json.dump({"Emission Probabilites":emission_probabilites}, outfile)


# In[70]:


with open("dev","r") as devfile:
    corpus = devfile.readlines()
dev_corpus = [sentence.lower() for sentence in corpus]


# In[71]:


with open("test","r") as devfile:
    corpus = devfile.readlines()
test_corpus = [sentence.lower() for sentence in corpus]


# In[110]:


cleaned_dev_corpus = []
sentence = []
for words in dev_corpus:
    if words[0]!='\n':
        word_tag = words.split("\t")
        word = (word_tag[1],word_tag[2].split('\n')[0])
        if word[0] in vocab:
            sentence.append(word)
        else:
            key = pos_tag([word[0]])[0][1]
            sentence.append(("<unk_"+key+">", word[1]))
    else:
        cleaned_dev_corpus.append(sentence)
        sentence = []
cleaned_dev_corpus.append(sentence)


# In[111]:


cleaned_test_corpus = []
sentence = []
for words in test_corpus:
    if words[0]=='\n':
        cleaned_test_corpus.append(sentence)
        sentence = []
    else:
        word_tag = words.split("\t")
        word = word_tag[1].split('\n')[0]
        if word in vocab:
            sentence.append(word)
        else:
            key = pos_tag([word])[0][1]
            sentence.append("<unk_"+key+">")
cleaned_test_corpus.append(sentence)


# In[74]:


def GreedyDecoding(sentence,flag):
    Tagged_POS = []
    for i in range(len(sentence)):
        if flag=="dev":
            word_to_be_tagged = sentence[i][0]
        else:
            word_to_be_tagged = sentence[i]
        #print(word_to_be_tagged)
#         if word_to_be_tagged not in vocab:
#             word_to_be_tagged = "<unk>"
        #print(word_to_be_tagged)
        max_value = 0
        max_POS = 0
        if(i==0):
            for POS,value in emisson_matrix[word_to_be_tagged].items():
                if POS in transition_matrix['.'].keys():
                    total_probability = value * transition_matrix['.'][POS]
                    if total_probability>=max_value:
                        max_value = total_probability
                        max_POS = POS
            if max_POS == 0:
                max_POS = max(transition_matrix['.'], key=transition_matrix['.'].get)
                #max_POS = random.choice(list(transition_matrix['.'].keys()))
                    
        else:
            for POS,value in emisson_matrix[word_to_be_tagged].items():
                #print(transition_matrix.keys())
                #print(Tagged_POS[i-1])
                if POS in transition_matrix[Tagged_POS[i-1]].keys():
                    total_probability = value * transition_matrix[Tagged_POS[i-1]][POS]
                    if total_probability>=max_value:
                        max_value = total_probability
                        max_POS = POS
            if max_POS == 0:
                max_POS = max(transition_matrix[Tagged_POS[i-1]], key=transition_matrix[Tagged_POS[i-1]].get)
                #max_POS = random.choice(list(transition_matrix[Tagged_POS[i-1]].keys()))
        Tagged_POS.append(max_POS)
        #print(Tagged_POS)
    return(Tagged_POS)
            


# In[75]:


def accuracy_model(data, algo):
    correct = 0
    total = 0
    output = []
    for sentence in data:
        Tagged_POS = algo(sentence,"dev")
        Actual_POS = [word[1] for word in sentence]
        Actual_sentence = [word[0] for word in sentence]
        correct_list = [1 if Tagged_POS[i]==Actual_POS[i] else 0 for i in range(len(Tagged_POS))]
        sum(correct_list)
        correct += sum(correct_list)
        total += len(correct_list)
        output.append(zip(Actual_sentence,Tagged_POS))
    acc = correct/total
    acc = acc * 100
    return acc,output


# In[76]:


def get_tags(data,algo):
    output = []
    for sentence in data:
        Tagged_POS = algo(sentence,"test")
        output.append([sentence,Tagged_POS])
    return output


# In[112]:


accuracy_greedy_dev, output_greed_dev = accuracy_model(cleaned_dev_corpus,GreedyDecoding)
print(accuracy_greedy_dev)


# In[113]:


output_greedy_test = get_tags(cleaned_test_corpus, GreedyDecoding)


# In[114]:


with open("greedy.out", "w") as outfile:
    for combined_sentence in output_greedy_test:
        #print(*sentence)
        index = 1
        for i in range(len(combined_sentence[0])):
            outfile.write(str(index)+ "\t" + combined_sentence[0][i] + "\t" + combined_sentence[1][i] + "\n")
            index+=1
        outfile.write("\n")


# In[80]:


def Viterbi(sentence,flag):
    V=[{}]
    for st in state_counts.keys():
#         print(sentence[0][0])
#         print(st)
        if flag=="dev":
            if st in emisson_matrix[sentence[0][0]].keys():
                emission_probability = emisson_matrix[sentence[0][0]][st]
            else:
                emission_probability = 0
        else:
            if st in emisson_matrix[sentence[0]].keys():
                emission_probability = emisson_matrix[sentence[0]][st]
            else:
                emission_probability = 0

        if st in transition_matrix['.'].keys():
            transition_probability = transition_matrix['.'][st]
        else:
            transition_probability = 0
#         print(transition_matrix['.'][st])
#         print(emisson_matrix[sentence[0][0]][st])
        V[0][st] = {"prob": emission_probability * transition_probability, "prev": None}
    
    for t in range(1, len(sentence)):
        V.append({})
        non_zero_initial_state = len(state_counts.keys())
        for st in state_counts.keys():
            keys = list(state_counts.keys())
            
            if st in transition_matrix[keys[0]]:
                initial_transition_probability = transition_matrix[keys[0]][st]
            else:
                initial_transition_probability = 0
                
            max_tr_prob = V[t - 1][keys[0]]["prob"] * initial_transition_probability
            prev_st_selected = keys[0]
            for prev_st in keys[1:]:
                #print(prev_st, st)
                if st in transition_matrix[prev_st]:
                    transition_probability = transition_matrix[prev_st][st]
                else:
                    transition_probability = 0
                #print(transition_matrix[prev_st][st])
                tr_prob = V[t - 1][prev_st]["prob"] * transition_probability
                if tr_prob > max_tr_prob:
                    max_tr_prob = tr_prob
                    prev_st_selected = prev_st
                                                      
            if flag=="dev":        
                if st in emisson_matrix[sentence[t][0]]:
                    emission_probability = emisson_matrix[sentence[t][0]][st]
                else:
                    emission_probability = 0
            else:
                if st in emisson_matrix[sentence[t]]:
                    emission_probability = emisson_matrix[sentence[t]][st]
                else:
                    emission_probability = 0

            max_prob = max_tr_prob * emission_probability
            V[t][st] = {"prob": max_prob, "prev": prev_st_selected}
            if max_prob!=0:
                non_zero_initial_state -= 1
                
        #print(non_zero_initial_state)
        if non_zero_initial_state == len(state_counts.keys()):
            prob = list(V[t-1].values())
            #previous_probabilities = print(prob[0]['prob'])
            previous_probabilities = [value['prob'] for value in prob]
            previous_state = [value['prev'] for value in prob]
            previous_max = max(previous_probabilities)
            prev_max_st_index = previous_probabilities.index(previous_max)
            for st in V[t].keys():
                V[t][st] = {"prob": previous_max, "prev": previous_state[prev_max_st_index]}
        
    opt = []
    max_prob = 0.0
    best_st = None
    #print(len(V))
    #print(sentence[15][0])
    #print(V[15].values())
    for st, data in V[-1].items():
        if data["prob"] > max_prob:
            max_prob = data["prob"]
            best_st = st
    opt.append(best_st)
    previous = best_st

    for t in range(len(V) - 2, -1, -1):
        opt.insert(0, V[t + 1][previous]["prev"])
        previous = V[t + 1][previous]["prev"]
    
    return opt


# In[ ]:


accuracy_greedy_test, output_greed_test = accuracy_model(cleaned_dev_corpus, Viterbi)
print(accuracy_greedy_test)


# In[ ]:


output_viterbi_test = get_tags(cleaned_test_corpus, Viterbi)


# In[ ]:


with open("viterbi.out", "w") as outfile:
    for combined_sentence in output_greedy_test:
        #print(*sentence)
        index = 1
        for i in range(len(combined_sentence[0])):
            outfile.write(str(index)+ "\t" + combined_sentence[0][i] + "\t" + combined_sentence[1][i] + "\n")
            index+=1
        outfile.write("\n")


# In[ ]:





# In[ ]:




