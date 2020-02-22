# Question 2: Changing self-loop weights


# This version of create phones' wfst allow for weights between phone substates to be changed on the base of a weight dictionary
# It is used also in the various word generator functions, as weights can also not be provided and the phone generator will use the default ones


def generate_phone_wfst_no_output(f, start_state, phone, n, counter, word_len, word, weight_dictionary):
    """
    Generate a WFST representating an n-state left-to-right phone HMM
    
    Args:
        f (fst.Fst()): an FST object, assumed to exist already
        start_state (int): the index of the first state, assmed to exist already
        phone (str): the phone label 
        n (int): number of states for each phone HMM
        
    Returns:
        the final state of the FST
    """
    if word!='sil':
        
        current_state = start_state
        for i in range(1, n+1):
            in_label = state_table.find('{}_{}'.format(phone, i))
            try:
                sl_weight = fst.Weight('log', -math.log(weight_dictionary[str(current_state)+str(current_state)]))
            except KeyError:
                sl_weight = fst.Weight('log', -math.log(weight_dictionary['self-loop']))  # weight for self-loop
                # self-loop back to current state
            f.add_arc(current_state, fst.Arc(in_label, 0, sl_weight, current_state))
        
            # transition to next state
        
            # we want to output the phone label on the final state
            # note: if outputting words instead this code should be modified
            if i == n and counter==word_len:
                out_label = word_table.find(word)
            else:
                out_label = 0   # output empty <eps> label
            
            next_state = f.add_state()
            try:
                next_weight = fst.Weight('log', -math.log(weight_dictionary[str(current_state)+str(next_state)]))
            except KeyError:
                next_weight = fst.Weight('log', -math.log(weight_dictionary['next'])) # weight to next state
            f.add_arc(current_state, fst.Arc(in_label, out_label, next_weight, next_state))    
       
            current_state = next_state
    else:
        # following code creates the wfst for silence model, a five state model having states 2,3,4 ergodically connected
        ergodic_states = {} # stores information for the ergodic states
        self_weight = fst.Weight("log",-math.log(0.1)) # Self-loop probability
        next_weight = fst.Weight("log",-math.log(0.9)) # Next state transition probability
        current_state = start_state
        for i in range(1,6):
            # fill ergodic_states dictionary
            if 1<i<5:
                ergodic_states[current_state] = state_table.find('{}_{}'.format('sil',i))
                next_state = f.add_state()
                if i==4:
                    # State 4 has 4 possible transitions: self-loop, state 1, state 2 and state 5. They need to sum up to 1 (self-loop=0.1, other transitions: uniformly distributed --> (1-0.1)/3 = 0.3
                    f.add_arc(current_state, fst.Arc(in_label, 0, fst.Weight("log",-math.log(0.3)), next_state))
                current_state = next_state
            # state 1 and 5 behaves as normal left to right wfst
            else:
                in_label = state_table.find('{}_{}'.format('sil',i))
                f.add_arc(current_state, fst.Arc(in_label,0,self_weight,current_state))
                next_state = f.add_state()
                f.add_arc(current_state, fst.Arc(in_label, 0, next_weight, next_state))
                current_state = next_state
        # add ergodic connections for states 2,3,4
        for key in ergodic_states.keys():
                for key2 in ergodic_states.keys():
                    if key==key2:
                        f.add_arc(key, fst.Arc(ergodic_states[key],0,self_weight,key))
                    elif ergodic_states[key]==state_table.find('sil_4'):
                        # See above at i==4 condition
                        f.add_arc(key, fst.Arc(ergodic_states[key],0,fst.Weight("log",-math.log(0.3)),key2))
                    else:
                        # All transitions need to sum up to 1, self-loop = 0.1, ergodic transitions: uniformly distributed--> (1-0.1)/2=0.45
                        f.add_arc(key, fst.Arc(ergodic_states[key],0,fst.Weight("log",-math.log(0.45)),key2))
            
    return current_state

# Question 2: unigram probabilities





def generate_multiple_words_wfst_unigram(word_list, unigram_probabilities, weight_dictionary):
    """ Generate a WFST for any word in the lexicon, composed of 3-state phone WFSTs.
        This will currently output word labels.  
        Exercise: could you modify this function and the one above to output a single phone label instead?
    
    Args:
        word (str): the word to generate
        
    Returns:
        the constructed WFST
    
    """
    if isinstance(word_list, str):
        word_list = word_list.split()
    f = fst.Fst("log")
    start_state = f.add_state()
    f.set_start(start_state)
    for word in word_list:
        # create the start state
        if word=='sil':
            
            current_state = f.add_state()
            
            f.add_arc(start_state, fst.Arc(0,0, fst.Weight("log",-math.log(0.1)), current_state)) # Probability to have a silence might be changed
            
            
            # If word is silence: apply special silence topology defined in generate_phone_wfst_no_output()
            # Arguments phone, n, word_len and weight_dictionary are obsolete in this case, they simply won't be used
            current_state = generate_phone_wfst_no_output(f, current_state, phone, 3, counter,len(lex[word]),word, weight_dictionary)
            
            f.add_arc(current_state, fst.Arc(0, 0, fst.Weight("log",-math.log(1)), start_state))
            
            f.set_final(current_state)
            
        
        else:
            
            current_state = f.add_state()
        
            f.add_arc(start_state, fst.Arc(0, 0, fst.Weight("log",-math.log(unigram_probabilities[word])), current_state))
        
            counter = 1
    
            # iterate over all the phones in the word
            for phone in lex[word]:   # will raise an exception if word is not in the lexicon
            
                current_state = generate_phone_wfst_no_output(f, current_state, phone, 3, counter,len(lex[word]),word, weight_dictionary)
            
                counter += 1
    
            # note: new current_state is now set to the final state of the previous phone WFST
        
            f.add_arc(current_state, fst.Arc(0, 0, fst.Weight("log",-math.log(1)), start_state))
    
            f.set_final(current_state)
        
    
    return f



# Define the transcription reader and create unigram_probability dictionary

def read_transcription(wav_file):
    """
    Get the transcription corresponding to wav_file.
    """
    
    transcription_file = os.path.splitext(wav_file)[0] + '.txt'
    
    with open(transcription_file, 'r') as f:
        transcription = f.readline().strip()
    
    return transcription

def create_unigram_probabilities(n=None):
    unigram_counts = {}
    tot = 0
    if n==None:
        for wav_file in glob.glob('/group/teaching/asr/labs/recordings/*.wav'):
            for word in read_transcription(wav_file).split():
                tot+=1
                if word in unigram_counts:
                    unigram_counts[word] += 1
                else:
                    unigram_counts[word] = 1
    else:
        counter = 0
        for wav_file in glob.glob('/group/teaching/asr/labs/recordings/*.wav'):
            if counter == n:
                break
            for word in read_transcription(wav_file).split():
                tot+=1
                if word in unigram_counts:
                    unigram_counts[word] += 1
                else:
                    unigram_counts[word] = 1
            counter += 1
    # Compute unigram probability for each word, discounting it uniformly by silence probability (0.1/len(unigram_counts))
    unigram_probability = {k:((v/tot)-(0.1/len(unigram_counts))) for k,v in unigram_counts.items()}
    return unigram_probability

unigram_probability = create_unigram_probabilities()
print(unigram_probability)


# Create the wfst (weight dictionary is empty by default, so if not passed, initial weights are used)
def create_wfst_unigram(unigram_probabilities, weight_dictionary={'self-loop':0.1,'next':0.9}):
    f = generate_multiple_words_wfst_unigram([k for k in lex.keys()], unigram_probabilities, weight_dictionary)
    f.set_input_symbols(state_table)
    f.set_output_symbols(word_table)
    return f


# Try it

f = create_wfst_unigram(unigram_probability)
f




# Question 4: Bigram Probability WFST

   
def generate_multiple_words_wfst_bigrams(word_list, weight_dictionary, bigram_dict):
    """ Generate a WFST for any word in the lexicon, composed of 3-state phone WFSTs.
        This will currently output word labels.  
        Exercise: could you modify this function and the one above to output a single phone label instead?
    
    Args:
        word (str): the word to generate
        
    Returns:
        the constructed WFST
    
    """
    if isinstance(word_list, str):
        word_list = word_list.split()
    f = fst.Fst("log")
    start_state = f.add_state()
    f.set_start(start_state)
    second_starts = {}
    ends = {}
    for word in word_list:
        if word!= 'sil':
            # create the start state
        
            current_state = f.add_state()
        
            f.add_arc(start_state, fst.Arc(0, 0, fst.Weight("log",-math.log(bigram_dict['<s>'+'/'+word])), current_state))
        
            counter = 1
    
            # iterate over all the phones in the word
            for phone in lex[word]:   # will raise an exception if word is not in the lexicon
        
                current_state = generate_phone_wfst_no_output(f, current_state, phone, 3, counter,len(lex[word]),word, weight_dictionary)
            
                counter += 1
    
                # note: new current_state is now set to the final state of the previous phone WFST
        
            f.set_final(current_state)
        
            second_start_state = current_state
            second_starts[word] = second_start_state
        
            for word2 in word_list:
                
                if word2 != 'sil':
            
                    current_state = f.add_state()
        
                    f.add_arc(second_start_state, fst.Arc(0, 0, fst.Weight("log",-math.log(bigram_dict[word+'/'+word2])), current_state))
        
                    counter = 1
    
                    # iterate over all the phones in the word
                    for phone in lex[word2]:   # will raise an exception if word is not in the lexicon
        
                        current_state = generate_phone_wfst_no_output(f, current_state, phone, 3, counter,len(lex[word2]),word2, weight_dictionary)
            
                        counter += 1
    
                    # note: new current_state is now set to the final state of the previous phone WFST
            
                    if word2 not in ends:
                        ends[word2] = [current_state]
                    else:
                        ends[word2] += [current_state]
                
                else:
                    
                    current_state = f.add_state()
                    
                    f.add_arc(second_start_state, fst.Arc(0,0,fst.Weight("log",-math.log(0.1)),current_state))
            
                    current_state = generate_phone_wfst_no_output(f, current_state, phone, 3, counter,len(lex[word]),word2, weight_dictionary)
            
                    f.add_arc(current_state, fst.Arc(0,0,fst.Weight("log",-math.log(1)),second_start_state))
                    
                    
        
        else:
            current_state = f.add_state()
            
            f.add_arc(start_state, fst.Arc(0,0,fst.Weight("log",-math.log(0.1)),current_state))
            
            current_state = generate_phone_wfst_no_output(f, current_state, phone, 3, counter,len(lex[word]),word, weight_dictionary)
            
            f.add_arc(current_state, fst.Arc(0,0,fst.Weight("log",-math.log(1)),start_state))
            
    for key in ends:
        for end in ends[key]:
            f.add_arc(end, fst.Arc(0, 0, fst.Weight("log",0), second_starts[key]))
    
    return f
 

def create_bigram_probabilities():
    unigram_counts = {'<s>':0}
    bigram_counts = {}
    for wav_file in glob.glob('/group/teaching/asr/labs/recordings/*.wav'):
        unigram_counts['<s>']+=1
        transcription = read_transcription(wav_file).split()
        for index,word in enumerate(transcription):
            if word in unigram_counts:
                unigram_counts[word] += 1
            else:
                unigram_counts[word] = 1
            if index==0:
                try:
                    bigram_counts['<s>'+'/'+word]+=1
                except KeyError:
                    bigram_counts['<s>'+'/'+word]=1
            elif index==len(transcription)-1:
                try:
                    bigram_counts[transcription[index-1]+'/'+word] += 1
                except KeyError:
                    bigram_counts[transcription[index-1]+'/'+word] = 1
                try:
                    bigram_counts[word+'/'+'<end>'] += 1
                except KeyError:
                    bigram_counts[word+'/'+'<end>'] = 1
            else:
                try:
                    bigram_counts[transcription[index-1]+'/'+word] += 1
                except KeyError:
                    bigram_counts[transcription[index-1]+'/'+word] = 1
    
    bigram_dict = {k:v/unigram_counts[k.split('/')[0]] for k,v in bigram_counts.items()}
            
    return bigram_dict

bigram_probability = create_bigram_probabilities()
print(bigram_probability)

# Try with 2 words to check the result

def create_wfst_bigrams(bigram_probability, weight_dictionary={}):
    f = generate_multiple_words_wfst_bigrams([k for k in lex.keys()], weight_dictionary, bigram_probability)
    f.set_input_symbols(state_table)
    f.set_output_symbols(word_table)
    return f

def create_wfst_bigrams_try(bigram_probability, weight_dictionary={'self-loop':0.1,'next':0.9}):
    f = generate_multiple_words_wfst_bigrams(['a','of','sil'], weight_dictionary, bigram_probability)
    f.set_input_symbols(state_table)
    f.set_output_symbols(word_table)
    return f

f = create_wfst_bigrams_try(bigram_probability)
f

# Looks like it is working!
# Eventually could be worth to think if we should include the probability of ending in a particular state (i.e. <end> given last word)
