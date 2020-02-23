__author__ = "s1735938, s1960659"
import observation_model
import math
import numpy as np
import glob
import os
import wer
import observation_model
import openfst_python as fst
from tqdm import tqdm_notebook as tqdm
import time

def read_transcription(wav_file):
    """
    Get the transcription corresponding to wav_file.
    """
    
    transcription_file = os.path.splitext(wav_file)[0] + '.txt'
    
    with open(transcription_file, 'r') as f:
        transcription = f.readline().strip()
    
    return transcription

def run_exp(wfst,num_test,beam_width=1e10,verbose=False):
    '''
    Run a test on the test data, record the WER, speed and memory cost.
    param wfst (pywrapfst._MutableFst)
    param num_test (int) The number of tests for the experiment. num_test=0 means using the dummy test.
    param prob_beam_width (float) The pruning beam width in negative log probability. By default corresponds to negative log 0.
    '''
    f = wfst
    
    # store the error counts and word counts
    tot_errors,tot_words,computation_counter = 0,0,0
    
    num_audio = len(glob.glob('/group/teaching/asr/labs/recordings/*.wav'))
    
    # take all if num_test is None
    dummy_test = False
    if num_test==0:
        print("dummy test")
        num_test = 1
        dummy_test = True
        
        
    # progress bar
    with tqdm(total=num_test) as progressbar:
        
        # start a timer
        start = time.time()
        
        for wav_file in glob.glob('/group/teaching/asr/labs/recordings/*.wav')[:num_test]:
            # update progress bar
            progressbar.update(1)
            
            if dummy_test:
                wav_file = ''
                transcription = "peppers"
            else:
                transcription = read_transcription(wav_file)
            
            # decoder.om.load_audio(wav_file)
            decoder = MyViterbiDecoder(f, wav_file)

            decoder.decode(beam_width = beam_width)
            (state_path, words) = decoder.backtrace()  # you'll need to modify the backtrace() from Lab 4
                                                       # to return the words along the best path

            # save the forward computation counter
            computation_counter += decoder.forward_counter
            
            if verbose:
                print(wav_file)
                print("recognized words: ",words)
                print("correct words: ", transcription)
            error_counts = wer.compute_alignment_errors(transcription, words)
            word_count = len(transcription.split())

            # increase the total error and word count
            tot_errors += np.sum(np.array(error_counts))
            tot_words += word_count
    
    # save the number states and arcs
    num_states,num_arcs = count_states_arcs(f)
    
    # stop the timer
    end = time.time()
    time_cost = end - start
    print("""
    Run time: {}, 
    Number of forward computations: {},
    Number of states and arcs: {} {},
    Number of errors {} ({}) in {} words .
    """.format(time_cost,computation_counter,num_states,num_arcs,tot_errors,error_counts,tot_words))     # you'll need to accumulate these to produce an overall Word Error Rate
    return time_cost,computation_counter,num_states,num_arcs,tot_errors,tot_words
        
        
def count_states_arcs (f):
    '''
    Count the number of states and arcs in an wfst
    type f pywrapfst
    para f the wfst to count the number of states and arcs for
    '''
    num_states = 0
    num_arcs = 0
    
    for state in f.states():
        num_states += 1

        for arc in f.arcs(state):
            num_arcs += 1

            
    return num_states,num_arcs


class MyWFST:
    
    def __init__(self,lexicon='lexicon.txt'):
        self.parse_lexicon(lexicon)
        self.generate_symbol_tables()
        
    def create_wfst_word_output(self, lm=None, tree_struc=False, weight_push=False, weight_dictionary={'self-loop':0.1, 'next':0.9}):
        '''
        wfst with word output
        '''
        f = self.generate_multiple_words_wfst_word_output([k for k in self.lex.keys()], weight_dictionary)
        f.set_input_symbols(self.state_table)
        f.set_output_symbols(self.word_table)
        
        if tree_struc:
            f = fst.determinize(f)
        if weight_push:
            f = f.push()
        return f

    def create_wfst(self):
        '''
        wfst with phone output
        '''
        self.parse_lexicon('lexicon.txt')
        self.generate_symbol_tables()

        f = self.generate_multiple_words_wfst([k for k in lex.keys()])
        f.set_input_symbols(self.state_table)
        f.set_output_symbols(self.phone_table)
        return f
    
    def parse_lexicon(self,lex_file):
        """
        Parse the lexicon file and return it in dictionary form.

        Args:
            lex_file (str): filename of lexicon file with structure '<word> <phone1> <phone2>...'
                            eg. peppers p eh p er z

        Returns:
            lex (dict): dictionary mapping words to list of phones
        """
    
        self.lex = {}  # create a dictionary for the lexicon entries (this could be a problem with larger lexica)
        with open(lex_file, 'r') as f:
            for line in f:
                line = line.split()  # split at each space
                self.lex[line[0]] = line[1:]  # first field the word, the rest is the phones



    def generate_symbol_tables(self, n=3):
        '''
        Return word, phone and state symbol tables based on the supplied lexicon

        Args:
            lexicon (dict): lexicon to use, created from the parse_lexicon() function
            n (int): number of states for each phone HMM

        Returns:
            word_table (fst.SymbolTable): table of words
            phone_table (fst.SymbolTable): table of phones
            state_table (fst.SymbolTable): table of HMM phone-state IDs
        '''

        self.state_table = fst.SymbolTable()
        self.phone_table = fst.SymbolTable()
        self.word_table = fst.SymbolTable()
    
        # add empty <eps> symbol to all tables
        self.state_table.add_symbol('<eps>')
        self.phone_table.add_symbol('<eps>')
        self.word_table.add_symbol('<eps>')
    
        for word, phones  in self.lex.items():
            if word=='sil':
                for i in range(1,6):
                    self.state_table.add_symbol('{}_{}'.format('sil',i))
        
            else:
                self.word_table.add_symbol(word)
        
                for p in phones: # for each phone
            
                    self.phone_table.add_symbol(p)
                    for i in range(1,n+1): # for each state 1 to n
                        self.state_table.add_symbol('{}_{}'.format(p, i))
            
        return self.word_table, self.phone_table, self.state_table
    
    def generate_multiple_words_wfst_word_output(self,word_list,weight_dictionary):
        """ Generate a WFST for any word in the lexicon, composed of 3-state phone WFSTs, where the last state of the last phone
        output the entire word.
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
        
            current_state = f.add_state()
        
            f.add_arc(start_state, fst.Arc(0, 0, fst.Weight("log",-math.log(1/len(word_list))), current_state))
            if word != 'sil':
                counter = 1
    
                # iterate over all the phones in the word
                for phone in self.lex[word]:   # will raise an exception if word is not in the lexicon
        
                    current_state = self.generate_phone_wfst_no_output(f, current_state, phone, 3, counter,len(self.lex[word]),word,   weight_dictionary)
            
                    counter += 1
    
                    # note: new current_state is now set to the final state of the previous phone WFST
        
                f.add_arc(current_state, fst.Arc(0, 0, fst.Weight("log",-math.log(1)), start_state)) # Can always
    
                f.set_final(current_state)
        
            else:
                # If word is silence: apply special silence topology defined in generate_phone_wfst_no_output()
                # Arguments phone, n, word_len and weight_dictionary are obsolete in this case, they simply won't be used
                current_state = self.generate_phone_wfst_no_output(f, current_state, phone, 3, counter,len(self.lex[word]),word, weight_dictionary)
            
                f.add_arc(current_state, fst.Arc(0, 0, fst.Weight("log",-math.log(1)), start_state))
    
                f.set_final(current_state)
    
        return f

    def generate_multiple_words_wfst(self,word_list):
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

            current_state = f.add_state()

            f.add_arc(start_state, fst.Arc(0, 0, fst.Weight("log",-math.log(1/len(word_list))), current_state))

            # iterate over all the phones in the word
            for phone in self.lex[word]:   # will raise an exception if word is not in the lexicon

                current_state = self.generate_phone_wfst(f, current_state, phone, 3)


                # note: new current_state is now set to the final state of the previous phone WFST

            f.add_arc(current_state, fst.Arc(0, 0, fst.Weight("log",-math.log(1)), start_state))

            f.set_final(current_state)

        return f
    
    def generate_phone_wfst_no_output(self, f, start_state, phone, n, counter, word_len, word, weight_dictionary):
        """
        Generate a WFST representating an n-state left-to-right phone HMM, but without outputting the phone at the final state

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
                in_label = self.state_table.find('{}_{}'.format(phone, i))
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
                    out_label = self.word_table.find(word)
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
                    ergodic_states[current_state] = self.state_table.find('{}_{}'.format('sil',i))
                    next_state = f.add_state()
                    if i==4:
                        # State 4 has 4 possible transitions: self-loop, state 1, state 2 and state 5. They need to sum up to 1 (self-loop=0.1, other transitions: uniformly distributed --> (1-0.1)/3 = 0.3
                        f.add_arc(current_state, fst.Arc(in_label, 0, fst.Weight("log",-math.log(0.3)), next_state))
                    current_state = next_state
                # state 1 and 5 behaves as normal left to right wfst
                else:
                    in_label = self.state_table.find('{}_{}'.format('sil',i))
                    f.add_arc(current_state, fst.Arc(in_label,0,self_weight,current_state))
                    next_state = f.add_state()
                    f.add_arc(current_state, fst.Arc(in_label, 0, next_weight, next_state))
                    current_state = next_state
            # add ergodic connections for states 2,3,4
            for key in ergodic_states.keys():
                for key2 in ergodic_states.keys():
                    if key==key2:
                        f.add_arc(key, fst.Arc(ergodic_states[key],0,self_weight,key))
                    elif ergodic_states[key]==self.state_table.find('sil_4'):
                        # See above at i==4 condition
                        f.add_arc(key, fst.Arc(ergodic_states[key],0,fst.Weight("log",-math.log(0.3)),key2))
                    else:
                        # All transitions need to sum up to 1, self-loop = 0.1, ergodic transitions: uniformly distributed--> (1-0.1)/2=0.45
                        f.add_arc(key, fst.Arc(ergodic_states[key],0,fst.Weight("log",-math.log(0.45)),key2))
            
        return current_state
     
    
    def generate_phone_wfst(self,f, start_state, phone, n):
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
    
        current_state = start_state

        for i in range(1, n+1):

            in_label = self.state_table.find('{}_{}'.format(phone, i))

            sl_weight = fst.Weight('log', -math.log(0.1))  # weight for self-loop
            # self-loop back to current state
            f.add_arc(current_state, fst.Arc(in_label, 0, sl_weight, current_state))

            # transition to next state

            # we want to output the phone label on the final state
            # note: if outputting words instead this code should be modified
            if i == n:
                out_label = self.phone_table.find(phone)
            else:
                out_label = 0   # output empty <eps> label

            next_state = f.add_state()
            next_weight = fst.Weight('log', -math.log(0.9)) # weight to next state
            f.add_arc(current_state, fst.Arc(in_label, out_label, next_weight, next_state))    

            current_state = next_state

        return current_state
    
    def generate_word_wfst(word):
        """ Generate a WFST for any word in the lexicon, composed of 3-state phone WFSTs.
            This will currently output word labels.  
            Exercise: could you modify this function and the one above to output a single phone label instead?

        Args:
            word (str): the word to generate

        Returns:
            the constructed WFST

        """
        f = fst.Fst('log')

        # create the start state
        start_state = f.add_state()
        f.set_start(start_state)

        current_state = start_state

        # iterate over all the phones in the word
        for phone in lex[word]:   # will raise an exception if word is not in the lexicon

            current_state = self.generate_phone_wfst(f, current_state, phone, 3)

            # note: new current_state is now set to the final state of the previous phone WFST

        f.set_final(current_state)

        return f
    
    def read_transcription(self,wav_file):
        """
        Get the transcription corresponding to wav_file.
        """
    
        transcription_file = os.path.splitext(wav_file)[0] + '.txt'
    
        with open(transcription_file, 'r') as f:
            transcription = f.readline().strip()
    
        return transcription
    
    def create_unigram_probabilities(self,n=None, sil_probability=0.1):
        unigram_counts = {}
        tot = 0
        if n==None:
            for wav_file in glob.glob('/group/teaching/asr/labs/recordings/*.wav'):
                for word in self.read_transcription(wav_file).split():
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
                for word in self.read_transcription(wav_file).split():
                    tot+=1
                    if word in unigram_counts:
                        unigram_counts[word] += 1
                    else:
                        unigram_counts[word] = 1
                counter += 1
        # Compute unigram probability for each word, discounting it uniformly by silence probability (0.1/len(unigram_counts))
        self.unigram_probability = {k:((v/tot)-(sil_probability/len(unigram_counts))) for k,v in unigram_counts.items()}
        return self.unigram_probability
    
    def generate_multiple_words_wfst_unigram(self, word_list, unigram_probabilities, weight_dictionary):
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
                current_state = self.generate_phone_wfst_no_output(f, current_state, phone, 3, counter,len(self.lex[word]),word, weight_dictionary)
            
                f.add_arc(current_state, fst.Arc(0, 0, fst.Weight("log",-math.log(1)), start_state))
            
                f.set_final(current_state)
            
        
            else:
            
                current_state = f.add_state()
        
                f.add_arc(start_state, fst.Arc(0, 0, fst.Weight("log",-math.log(unigram_probabilities[word])), current_state))
        
                counter = 1
    
                # iterate over all the phones in the word
                for phone in self.lex[word]:   # will raise an exception if word is not in the lexicon
            
                    current_state = self.generate_phone_wfst_no_output(f, current_state, phone, 3, counter,len(self.lex[word]),word, weight_dictionary)
            
                    counter += 1
    
                # note: new current_state is now set to the final state of the previous phone WFST
        
                f.add_arc(current_state, fst.Arc(0, 0, fst.Weight("log",-math.log(1)), start_state))
    
                f.set_final(current_state)
        
    
        return f
    
    
    def create_wfst_unigram(self, lm=None, tree_struc=False, weight_push=False, weight_dictionary={'self-loop':0.1,'next':0.9}):
        f = self.generate_multiple_words_wfst_unigram([k for k in self.lex.keys()], self.unigram_probability, weight_dictionary)
        f.set_input_symbols(self.state_table)
        f.set_output_symbols(self.word_table)
        if tree_struc:
            f = fst.determinize(f)
        if weight_push:
            f = f.push()
        return f
    
    
    def create_bigram_probabilities(self, n=None, sil_probability=0.1):
        unigram_counts = {'<s>':0}
        bigram_counts = {}
        if n==None:
            for wav_file in glob.glob('/group/teaching/asr/labs/recordings/*.wav'):
                unigram_counts['<s>']+=1
                transcription = self.read_transcription(wav_file).split()
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
        else:
            counter = 0
            for wav_file in glob.glob('/group/teaching/asr/labs/recordings/*.wav'):
                if counter == n:
                    break
                unigram_counts['<s>']+=1
                transcription = self.read_transcription(wav_file).split()
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
                    counter += 1
            
        # Create bigram probabilities from accumulated counts, discounting uniformly 0.1 for silence (the value can be changed)
        self.bigram_probability = {k:(v/unigram_counts[k.split('/')[0]])-(sil_probability/len(unigram_counts)) for k,v in bigram_counts.items()}
            
        return self.bigram_probability
    
    def generate_multiple_words_wfst_bigrams(self,word_list, weight_dictionary, bigram_dict):
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
                for phone in self.lex[word]:   # will raise an exception if word is not in the lexicon
        
                    current_state = self.generate_phone_wfst_no_output(f, current_state, phone, 3, counter,len(self.lex[word]),word, weight_dictionary)
            
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
                        for phone in self.lex[word2]:   # will raise an exception if word is not in the lexicon
        
                            current_state = self.generate_phone_wfst_no_output(f, current_state, phone, 3, counter,len(self.lex[word2]),word2, weight_dictionary)
            
                            counter += 1
    
                        # note: new current_state is now set to the final state of the previous phone WFST
            
                        if word2 not in ends:
                            ends[word2] = [current_state]
                        else:
                            ends[word2] += [current_state]
                
                    else:
                    
                        current_state = f.add_state()
                    
                        f.add_arc(second_start_state, fst.Arc(0,0,fst.Weight("log",-math.log(0.1)),current_state))
            
                        current_state = self.generate_phone_wfst_no_output(f, current_state, phone, 3, counter,len(self.lex[word]),word2, weight_dictionary)
            
                        f.add_arc(current_state, fst.Arc(0,0,fst.Weight("log",-math.log(1)),second_start_state))
                    
                    
        
            else:
                current_state = f.add_state()
            
                f.add_arc(start_state, fst.Arc(0,0,fst.Weight("log",-math.log(0.1)),current_state))
            
                current_state = self.generate_phone_wfst_no_output(f, current_state, phone, 3, counter,len(self.lex[word]),word, weight_dictionary)
            
                f.add_arc(current_state, fst.Arc(0,0,fst.Weight("log",-math.log(1)),start_state))
            
        for key in ends:
            for end in ends[key]:
                f.add_arc(end, fst.Arc(0, 0, fst.Weight("log",0), second_starts[key]))
    
        return f
    
    def create_wfst_bigrams(self, lm=None, tree_struc=False, weight_push=False, weight_dictionary={'self-loop':0.1,'next':0.9}):
        f = self.generate_multiple_words_wfst_bigrams([k for k in self.lex.keys()], weight_dictionary, self.bigram_probability)
        f.set_input_symbols(self.state_table)
        f.set_output_symbols(self.word_table)
        if tree_struc:
            f = fst.determinize(f)
        if weight_push:
            f = f.push()
        return f

    def create_wfst_bigrams_try(self, lm=None, tree_struc=False, weight_push=False, weight_dictionary={'self-loop':0.1,'next':0.9}):
        f = self.generate_multiple_words_wfst_bigrams(['a','of','sil'], weight_dictionary, self.bigram_probability)
        f.set_input_symbols(self.state_table)
        f.set_output_symbols(self.word_table)
        if tree_struc:
            f = fst.determinize(f)
        if weight_push:
            f = f.push()
        return f
    
    

class MyViterbiDecoder:
    
    NLL_ZERO = 1e10  # define a constant representing -log(0).  This is really infinite, but approximate
                     # it here with a very large number
    
    def __init__(self, f, audio_file_name):
        """Set up the decoder class with an audio file and WFST f
        """
        self.om = observation_model.ObservationModel()
        self.f = f
        self.forward_counter = 0

        if audio_file_name:
            self.om.load_audio(audio_file_name)
        else:
            self.om.load_dummy_audio()
        
        self.initialise_decoding()

        
    def initialise_decoding(self):
        """set up the values for V_j(0) (as negative log-likelihoods)
        
        """
        
        self.V = []   # stores likelihood along best path reaching state j
        self.B = []   # stores identity of best previous state reaching state j
        self.W = []   # stores output labels sequence along arc reaching j - this removes need for 
                      # extra code to read the output sequence along the best path
        
        for t in range(self.om.observation_length()+1):
            self.V.append([self.NLL_ZERO]*self.f.num_states())
            self.B.append([-1]*self.f.num_states())
            self.W.append([[] for i in range(self.f.num_states())])  #  multiplying the empty list doesn't make multiple
        
        # The above code means that self.V[t][j] for t = 0, ... T gives the Viterbi cost
        # of state j, time t (in negative log-likelihood form)
        # Initialising the costs to NLL_ZERO effectively means zero probability    
        
        # give the WFST start state a probability of 1.0   (NLL = 0.0)
        self.V[0][self.f.start()] = 0.0
        
        # some WFSTs might have arcs with epsilon on the input (you might have already created 
        # examples of these in earlier labs) these correspond to non-emitting states, 
        # which means that we need to process them without stepping forward in time.  
        # Don't worry too much about this!  
        self.traverse_epsilon_arcs(0)        
        
    def traverse_epsilon_arcs(self, t):
        """Traverse arcs with <eps> on the input at time t
        
        These correspond to transitions that don't emit an observation
        
        We've implemented this function for you as it's slightly trickier than
        the normal case.  You might like to look at it to see what's going on, but
        don't worry if you can't fully follow it.
        
        """
        
        states_to_traverse = list(self.f.states()) # traverse all states
        while states_to_traverse:
            
            # Set i to the ID of the current state, the first 
            # item in the list (and remove it from the list)
            i = states_to_traverse.pop(0)   
        
            # don't bother traversing states which have zero probability
            if self.V[t][i] == self.NLL_ZERO:
                    continue
        
            for arc in self.f.arcs(i):
                
                if arc.ilabel == 0:     # if <eps> transition
                  
                    j = arc.nextstate   # ID of next state  
                
                    if self.V[t][j] > self.V[t][i] + float(arc.weight):
                        
                        self.forward_counter += 1 # increase the forward counter by 1
                        
                        # this means we've found a lower-cost path to
                        # state j at time t.  We might need to add it
                        # back to the processing queue.
                        self.V[t][j] = self.V[t][i] + float(arc.weight)
                        
                        # save backtrace information.  In the case of an epsilon transition, 
                        # we save the identity of the best state at t-1.  This means we may not
                        # be able to fully recover the best path, but to do otherwise would
                        # require a more complicated way of storing backtrace information
                        self.B[t][j] = self.B[t][i] 
                        
                        # and save the output labels encountered - this is a list, because
                        # there could be multiple output labels (in the case of <eps> arcs)
                        if arc.olabel != 0:
                            # not executed so far
                            self.W[t][j] = self.W[t][i] + [arc.olabel]
                        else:
#                             print("traverse epsilon j:", self.W[t][j])
#                             print("traverse epsilon i:", self.W[t][i])
                             self.W[t][j] = self.W[t][i]
                        
                        if j not in states_to_traverse:
                            states_to_traverse.append(j)

    
    def forward_step(self, t):
        
        # init best_cost, which will be used in beam search. The cost larger than `best_cost` + `beam_width` (for pruning).
        best_cost = self.NLL_ZERO
        
        for i in self.f.states():
            
            if not self.V[t-1][i] == self.NLL_ZERO:   # no point in propagating states with zero probability
                
                for arc in self.f.arcs(i):
                    
                    if arc.ilabel != 0: # <eps> transitions don't emit and observation
                        
                        self.forward_counter += 1 # increase the forward counter by 1
                        j = arc.nextstate
                        tp = float(arc.weight)  # transition prob
                        ep = -self.om.log_observation_probability(self.f.input_symbols().find(arc.ilabel), t)  # emission negative log prob
                        prob = tp + ep + self.V[t-1][i] # they're logs
                        
                        # if the nega logprob is larger than the lowest at this time step plus the beam width, does nothing.
                        if prob > best_cost + self.beam_width:
                            continue
                        
                        # else if it is lower than the current viterbi value at time t state j, update the viterbi value and write the backpointer...
                        elif prob < self.V[t][j]:
                            self.V[t][j] = prob
                            self.B[t][j] = i
                            
                            # update the BEST_COST
                            best_cost = prob
                            
                            # store the output labels encountered too
                            if arc.olabel !=0:
                                self.W[t][j] = [arc.olabel]
                            else:
                                # not executed so far
                                self.W[t][j] = []
                            
    
    def finalise_decoding(self):
        """ this incorporates the probability of terminating at each state
        """
        
        for state in self.f.states():
            final_weight = float(self.f.final(state))
            if self.V[-1][state] != self.NLL_ZERO:
                if final_weight == math.inf:
                    self.V[-1][state] = self.NLL_ZERO  # effectively says that we can't end in this state
                else:
                    self.V[-1][state] += final_weight
                    
        # get a list of all states where there was a path ending with non-zero probability
        finished = [x for x in self.V[-1] if x < self.NLL_ZERO]
        if not finished:  # if empty
            print("No path got to the end of the observations.")
        
        
    def decode(self, beam_width=0):
        self.initialise_decoding()
        t = 1
        # add instance variable: beam_width 
        self.beam_width=beam_width
        while t <= self.om.observation_length():
            self.forward_step(t)
            self.traverse_epsilon_arcs(t)
            t += 1
        self.finalise_decoding()
    
    def backtrace(self):
        
        best_final_state = self.V[-1].index(min(self.V[-1])) # argmin
        best_state_sequence = [best_final_state]
        best_out_sequence = []
        
        t = self.om.observation_length()   # ie T
        j = best_final_state
        prev_j = -1
        while t >= 0:
            i = self.B[t][j]
            best_state_sequence.append(i)

            best_out_sequence = self.W[t][j] + best_out_sequence  # computer scientists might like
                                                                                # to make this more efficient!

            # continue the backtrace at state i, time t-1
            j = i  
            t-=1
            
        best_state_sequence.reverse()
        
        # convert the best output sequence from FST integer labels into strings
        best_out_sequence = ' '.join([ self.f.output_symbols().find(label) for label in best_out_sequence])

        return (best_state_sequence, best_out_sequence)