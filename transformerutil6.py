"""
Copyright 2020 Aditya Gomatam.

This file is part of Music-Transformer (https://github.com/spectraldoy/Music-Transformer), my project to build and
train a Music Transformer. Music-Transformer is open-source software licensed under the terms of the GNU General
Public License v3.0. Music-Transformer is free software: you can redistribute it and/or modify it under the terms of
the GNU General Public License as published by the Free Software Foundation, either version 3 of the License,
or (at your option) any later version. Music-Transformer is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
See the GNU General Public License for more details. A copy of this license can be found within the GitHub repository
for Music-Transformer, or at https://www.gnu.org/licenses/gpl-3.0.html.
"""


import mido
import tensorflow as tf
import numpy as np
import random

"""
Implementation of a converter of MIDI files to and from the event-based
vocabulary representation of MIDI files according to Oore et al., 2018
Also some heloer fuctions to be able to use the transformer model properly

Possible MIDI events being considered:
    128 note_on events
    128 note_off events #includes handling pedal_on and pedal_off events
    125 time_shift events #time_shift = 1: 8 ms
    32  velocity events
    
Total number of midi events = 413

The indices of the vocab corresponding to the events will be,
v[       0] = '<pad>'
v[  0..128] = note_on
v[129..256] = note_off
v[257..381] = time_shift
v[382..413] = velocity
v[414..415] = '<start>', '<end>'

A list of tokens will be generated from the midi file, and the indices of these
need to be passed into the Embedding
"""

"""MIDI TOKENIZER"""
note_on_events = 128
note_off_events = note_on_events
time_shift_events = 125 #time_shift = time_shift_events corresponds to 1 second
velocity_events = 32

LTH = 1000 #maximum number of milliseconds to be handled
DIV = 1000 // time_shift_events #time_shifts will correspond to steps of DIV milliseconds

#total midi events has + pad + start + end
total_midi_events = note_on_events + note_off_events + time_shift_events + velocity_events + 1 + 2 

#create the vocabulary list to use when pedal is considered to be holding the 
#note,instead of introducing pedal events -- nn might be able to learn easier 
note_on_vocab = [f"note_on_{i}" for i in range(note_on_events)]
note_off_vocab = [f"note_off_{i}" for i in range(note_off_events)]
time_shift_vocab = [f"time_shift_{i}" for i in range(time_shift_events)]
velocity_vocab = [f"set_velocity_{i}" for i in range(velocity_events)]

#create vocab of tokens
vocab = ['<pad>'] + note_on_vocab + note_off_vocab + time_shift_vocab + velocity_vocab + ['<start>', '<end>']
vocab_size = len(vocab)

#tokena
pad_token = vocab.index("<pad>")
start_token = vocab.index("<start>")
end_token = vocab.index("<end>")
        
def Midiparser(fname=None, mid=None):
    """
    Converts a midi file into a list of events and their indices in the vocab
    """
    assert (fname == None) ^ (mid == None) == True, "Define only one of mid (a loaded midi file) or fname (the path from which to load a midi file)"
    if fname is not None:
        mid = mido.MidiFile(fname)

    #conversion macros
    delta_time = 0 #to be able to sum up message times
    event_list = [] #list of events from vocab list
    index_list = [] #list of indices of the events of event_list in the vocab
    pedal_events = {} #dictionary to handle pedal events
    pedal_flag = False
    
    #create the event list as a list of elements of the vocab
    for track in mid.tracks:
        for msg in track:
            delta_time += msg.time
            if msg.is_meta:
                continue
            
            #add the time events
            t = msg.type
            if t == "note_on" or (t == "note_off" and not pedal_flag):
                time_to_events(delta_time, event_list=event_list, index_list=index_list)
                delta_time = 0
                
            if t == "note_on":
                #get the note
                note = msg.note
                vel = velocity_to_bin(msg.velocity)
                
                #append the set velocity and note events; 1 is added to deal with the pad token
                event_list.append(vocab[note_on_events + note_off_events + time_shift_events + vel + 1])
                event_list.append(vocab[note + 1])
                
                index_list.append(note_on_events + note_off_events + time_shift_events + vel + 1)
                index_list.append(note + 1)
                
            elif t == "note_off" and not pedal_flag:
                #get the note
                note = msg.note
                
                #append the note off event
                event_list.append(vocab[note_on_events + note + 1])
                index_list.append(note_on_events + note + 1)
                
            elif t == "note_off" and pedal_flag:
                note = msg.note
                if note not in pedal_events:
                    pedal_events[note] = 0
                pedal_events[note] += 1
            
            elif msg.type == "control_change":
                if msg.control == 64:
                    if msg.value >= 64:
                        #pedal on
                        pedal_flag = True 
                    elif msg.value <= 63:
                        #pedal off
                        pedal_flag = False 
                        
                        #perform note offs that occurred when pedal was on, after the pedal is lifted
                        for note in pedal_events:
                            for i in range(pedal_events[note]):
                                #add the time events
                                time_to_events(delta_time, event_list=event_list, index_list=index_list)
                                delta_time = 0
                                #repeatedly create and append note off events
                                event_list.append(vocab[note_on_events + note + 1])
                                index_list.append(note_on_events + note + 1)
                        #restart the pedal events list
                        pedal_events = {}
    
    #return the lists of events
    return np.array(index_list, dtype=np.int32), event_list

def Listparser(index_list=None, event_list=None, fname="test", tempo=512820):
    """
    Takes a set of events in event_list or in index_list and converts it to a midi file
    """
    assert (event_list == None) ^ (index_list == None) == True, "Input either the event_list or index_list but not both"
    
    #convert event_list to index_list
    if event_list is not None:
        assert type(event_list[0]) == str, "All events in event_list must be str"
        index_list = events_to_indices(event_list)
    #set up the midi file and tracks to be added to it
    mid = mido.MidiFile() #ticks_per_beat should be 480
    meta_track = mido.MidiTrack()
    track = mido.MidiTrack()

    # set up the config track
    meta_track.append(mido.MetaMessage("track_name").copy(name=fname))
    meta_track.append(mido.MetaMessage("smpte_offset"))  # open track
    # time_signature
    time_sig = mido.MetaMessage("time_signature")  # assumes time sig is 4/4
    time_sig = time_sig.copy(numerator=4, denominator=4)
    meta_track.append(time_sig)
    # key signature
    key_sig = mido.MetaMessage("key_signature")  # assumes key sig is C
    meta_track.append(key_sig)
    # tempo
    set_tempo = mido.MetaMessage("set_tempo")  # assume tempo is constant
    set_tempo = set_tempo.copy(tempo=tempo)
    meta_track.append(set_tempo)
    # end of track
    end = mido.MetaMessage("end_of_track")
    end = end.copy(time=0)  # time is delta time
    meta_track.append(end)  # check if this is the isolated problem

    # set up the piano track
    program = mido.Message("program_change")  # 0 is piano
    track.append(program)
    # control
    cc = mido.Message("control_change")
    track.append(cc)  # looks like that's done
    
    #initialize the time and velocity attributes
    delta_time = 0
    vel = 0
    
    #iterate over the events in event list to reconstruct the midi file
    for idx in index_list:
        if tf.is_tensor(idx):
            idx = idx.numpy().item()
        if idx == 0: #if it is the pad token, continue
            continue
        idx = idx - 1 #subtracting 1 to deal with the pad token
        if 0 <= idx < note_on_events + note_off_events:
            if 0 <= idx < note_on_events:
                #note on event
                note = idx
                t = "note_on"
                v = vel
            else:
                #note off event
                note = idx - note_on_events
                t = "note_off"
                v = 127
            #set up the message
            msg = mido.Message(t)
            msg = msg.copy(note=note, velocity=v, time=delta_time)
            #reinitialize delta_time and velocity
            delta_time = 0
            vel = 0
            
            #insert message into track
            track.append(msg)
            
        elif note_on_events + note_off_events <= idx < note_on_events + note_off_events + time_shift_events:
            #time shift event
            cut_time = idx - (note_on_events + note_off_events - 1) # from 1 to time_shift_events 
            delta_time += cut_time * DIV #div is used to turn the time from bins to milliseconds
            
        elif note_on_events + note_off_events + time_shift_events <= idx < total_midi_events - 3: #subtract start and end tokens
            #velocity event
            vel = bin_to_velocity(idx - (note_on_events + note_off_events + time_shift_events))
    
    #end the track
    end2 = mido.MetaMessage("end_of_track").copy(time=0)
    track.append(end2)
    
    #create and return the midi file
    mid.tracks.append(meta_track)
    mid.tracks.append(track)
    return mid
    
def events_to_indices(event_list, vocab=vocab):
    """
    turns an event_list into an index_list
    """
    index_list = []
    for event in event_list:
        index_list.append(vocab.index(event))
    return tf.convert_to_tensor(index_list)

def indices_to_events(index_list, vocab=vocab):
    """
    turns an index_list into an event_list
    """
    event_list = []
    for idx in index_list:
        event_list.append(vocab[idx])
    return event_list

def velocity_to_bin(velocity, step=4):
    """
    Velocity in a midi file can take on any integer value in the range (0, 127)
    But, so that each vector in the midiparser is fewer dimensions than it has to be, 
    without really losing any resolution in dynamics, the velocity is shifted 
    down to the previous multiple of step
    """
    assert (128 % step == 0), "128 possible midi velocities must be divisible into the number of bins"
    assert 0 <= velocity <= 127, f"velocity must be between 0 and 127, not {velocity}"
    
    #bins = np.arange(0, 127, step) #bins[i] is the ith multiple of step, i.e., step * i
    idx = velocity // step
    return idx #returns the bin into which the actual velocity is placed

def bin_to_velocity(_bin, step=4):
    """
    Takes a binned velocity, i.e., a value from 0 to 31, and converts it to a 
    proper midi velocity
    """
    assert (0 <= _bin * step <= 127), f"bin * step must be between 0 and 127 to be a midi velocity\
                                        not {_bin*step}"
    
    return int(_bin * step)

def time_to_events(delta_time, event_list=None, index_list=None):
    """
    takes the delta time summed up over irrelevant midi events, and converts it
    to a series of keys from the vocab
    """
    #since msg.time is the time since the previous message, the time
    #shift events need to be put before the next messages
    time = time_cutter(delta_time)
    for i in time:
        #repeatedly create and append time events
        if event_list is not None:
            event_list.append(vocab[note_on_events + note_off_events + i])  #should be -1, but adding +1
        if index_list is not None:                                          #because of pad token
            index_list.append(note_on_events + note_off_events + i)
    pass
    

def time_cutter(time, lth=LTH, div=DIV):
    """
    In the mido files, with ticks_per_beat = 480, the default tempo
    is 480000 µs/beat or 125 bpm. This does not depend on the time signature
    1 tick is 1 ms at this tempo, therefore 8 ticks are 8 ms, which each of the
    bins for time are supposed to be
    
    lth is the maximum number of ticks, or milliseconds in this case, that will
    be considered in one time_shift for this project
    div is the number of milliseconds/ticks a time_shift of 1 represents, i.e.
    time = time_shift * div
    
    this function makes mido time attributes into multiplies of div, in the 
    integer range (1, lth); 0 will not be considered; then divides them into 
    lth // div possible bins, integers from 1 to lth // div
    
    """
    assert (lth % div == 0), "lth must be divisible by div"
   
    #create in the time shifts in terms of integers in the range (1, lth) then
    #convert the time shifts into multiples of div, so that we only need to deal with 
    #lth // div possible bins
    time_shifts = []
    
    for i in range(time // lth):
        time_shifts.append(real_round(lth / div)) #see below for real_round
    last_term = real_round((time % lth) / div)
    time_shifts.append(last_term) if last_term > 0 else None
    
    return time_shifts


def check_note_pairs(fname=None, mid=None, return_notes=False):
    """
    checks if each note_on is paired with a note_off in a midi file
    """
    assert (fname == None)^(mid == None) == True, "Define only one of mid (a loaded midi file) or fname (the path from which to load a midi file)"
    if fname is not None:
        mid = mido.MidiFile(fname)
    
    notes = {}
    for track in mid.tracks:
        for msg in track:
            if msg.is_meta or (msg.type != "note_on" and msg.type != "note_off"):
                continue
            note = msg.note
            t = msg.type
            if note not in notes:
                notes[note] = 0
                
            if t == "note_on":
                notes[note] += 1
            elif t == "note_off":
                notes[note] -= 1
    
    flag = True # all note pairs exist
    
    for i in notes:
        if notes[note] != 0:
            flag = False
            break
    if return_notes:
        return notes
    return flag #, notes

def real_round(a):
    """
    properly rounds a float to an integer because python can't do it
    """
    b = a // 1
    decimal_digits = a % 1
    adder = 0
    if decimal_digits >= 0.5:
        adder = 1
    return int(b + adder)

"""TRANSFORMER UTIL"""

MAX_LENGTH = 2048

#stuff to make the data augmentation easier
noe = note_on_events
nfe = note_off_events
ne = noe + nfe #note events
tse = time_shift_events

def skew(t: tf.Tensor):
    """
    Implements skewing procedure outlined in Huang et. al 2018 to reshape the
    dot(Q, RelativePositionEmbeddings) matrix into the correct shape for which
    Tij = compatibility of ith query in Q with relative position (j - i)
    
    this implementation accounts for tensors of rank n
    
    Algorithm:
        1. Pad T
        2. Reshape
        3. Slice
    
    Assumes T is of shape (..., L, L)
    """
    # pad T
    middle_dims = [[0, 0] for _ in range(tf.rank(t) - 1)] # allows padding to be generalized to rank n
    padded = tf.pad(t, [*middle_dims, [1, 0]])
    
    # reshape
    srel = tf.reshape(padded, (*padded.shape[:-2], t.shape[-1] + 1, t.shape[-2]))
    
    # final touches
    srel = tf.reshape(srel, (-1, *srel.shape[-2:])) # flatten prior dims
    srel = srel[:, 1:] # slice
    return tf.reshape(srel, t.shape) # prior shape

def data_cutter(data, lth=MAX_LENGTH):
    """
    takes a set of long input sequences, data, and cuts each sequence into 
    smaller sequences of length lth + 2
    the start and end tokens are also added to the data
    """
    #make sure data is iterable
    if type(data) != list:
        data = [data]
        
    #initialize the cut data list and seqs to pad to add later to the cut data
    cdata = []
    seqs_to_pad = []
    
    for seq in data:
        #find the highest multiple of lth less than len(seq) so that until
        #this point, the data can be cut into even multiples of lth
        seq_len = len(seq)
        if lth > seq_len:
            seqs_to_pad.append(seq)
            continue
            
        mult = seq_len // lth
        
        #iterate over parts of the sequence of length lth and add them to cdata
        for i in range(0, lth * (mult), lth):
            _slice = seq[i:i + lth]
            cdata.append(_slice)
        #take the last <lth elements of the sequnce and add to seqs_to_pad
        idx = mult * lth
        final_elems = seq[idx:]
        seqs_to_pad.append(final_elems) if final_elems.size > 0 else None
    
    #add the start and end tokens
    for i, vec in enumerate(cdata):
        # assume vec is of rank 1
        cdata[i] = tf.pad(tf.pad(vec, [[1, 0]], constant_values=start_token), \
             [[0, 1]], constant_values=end_token)
    for i, vec in enumerate(seqs_to_pad):
        seqs_to_pad[i] = tf.pad(tf.pad(vec, [[1, 0]], constant_values=start_token), \
             [[0, 1]], constant_values=end_token)
    
    #pad the sequences to pad
    if seqs_to_pad:
        padded_data = tf.keras.preprocessing.sequence.pad_sequences(seqs_to_pad, maxlen=lth + 2, 
                                                                    padding='post',value=pad_token)
        final_data = tf.concat([tf.expand_dims(cd, 0) for cd in cdata] + \
                                [tf.expand_dims(pd, 0) for pd in padded_data], 0)
    else:
        final_data = tf.concat([tf.expand_dims(cd, 0) for cd in cdata], 0)
    return final_data
    
def start_end_separator(data, lth=MAX_LENGTH):
    """
    function to return only the first and last lth tokens of the index lists in data
    as numpy arrays
    input index lists are assumed to be numpy arrays
    also pads the input data with start and end tokens
    """
    if type(data) != list:
        data = [data]
    
    sep_data = []
    seqs_to_pad = []
    
    # separate the data and append to correct lists
    for arr in data:
        if len(arr) == lth:
            sep_data.append(arr)
        elif len(arr) < lth:
            seqs_to_pad.append(arr)
        else:
            first = arr[:lth]
            last = arr[-lth:]
            sep_data.append(first)
            sep_data.append(last)
    
    # add start and end tokens
    for i, vec in enumerate(sep_data):
        # assume vec is of rank 1
        sep_data[i] = tf.pad(tf.pad(vec, [[1, 0]], constant_values=start_token), \
             [[0, 1]], constant_values=end_token)
    for i, vec in enumerate(seqs_to_pad):
        seqs_to_pad[i] = tf.pad(tf.pad(vec, [[1, 0]], constant_values=start_token), \
             [[0, 1]], constant_values=end_token)
    # pad seqs to pad
    padded_data = tf.keras.preprocessing.sequence.pad_sequences(seqs_to_pad, maxlen=lth + 2, 
                                                                padding='post', value=pad_token)
    
    # concatenate
    return tf.concat([tf.expand_dims(sd, 0) for sd in sep_data] + \
                      [tf.expand_dims(pd, 0) for pd in padded_data], 0)
    

def stretch_time(seq, time_stretch):
    """
    function to help data augmentation that stretches index list in time
    """
    # initialize time_shifted sequence to return
    time_shifted_seq = []
    delta_time = 0
    
    # iterate over seq
    if time_stretch == 1:
        if type(seq) == np.ndarray:
            return seq
        else:
            return np.array(seq)
    for idx in seq:
        idx = idx.item()
        #if idx is a time_shift
        if ne < idx <= ne + tse:
            time = idx - (ne - 1) # get the index in the vocab
            delta_time += real_round(time * DIV * time_stretch) #acculumate stretched times
        else:
            time_to_events(delta_time, index_list=time_shifted_seq) #add the accumulated stretched times to the list
            delta_time = 0 #reinitialize delta time
            time_shifted_seq.append(idx) #add other indices back
            
    return np.array(time_shifted_seq, dtype=np.int32) #np ndarray instead of tf tensor to save

def aug(data, note_shifts=np.arange(-2, 3), time_stretches=[1, 1.05, 1.1], 
        sample_random_time=False, sample_size=None):
    """
    uses note_shifts and time_stretches to implement the data augmentation
    on data, which is a set of sequences of index_lists defined by
    miditokenizer3
    assumes note_shifts are integers
    
    should put an assert positive for time stretches
    """
    assert type(note_shifts) == list or type(note_shifts) == np.ndarray, \
                                "note_shifts must be a list of integers(number of semitones) to shift pitch by"
    assert type(time_stretches) == list, "time_stretches must be a list of coefficients"
    
    assert (sample_random_time == True) ^ (sample_size is None), "Define none or both of sample_random_time and sample_size"
    assert (sample_size is None) or type(sample_size) == int, "sample_size must be an int"
    
    #make sure data is in a list
    if type(data) != list:
        data = [data]
    
    #preprocess the time stretches
    if 1 not in time_stretches:
        time_stretches.append(1)
    ts = []
    for t in time_stretches:
        ts.append(t)
        ts.append(1/t) if t != 1 else None
    ts.sort() #make it ascending
    
    predicted_len = len(data) * len(note_shifts) * sample_size if sample_random_time else len(data) * len(note_shifts) * len(ts)
    print(f'Predicted number of augmented data samples: {predicted_len}')
    
    #iterate over the sequences in the data to shift each one of them
    note_shifted_data = [] #initialize the set of note_shifted sequences
    count = 0
    
    for seq in data:
        #data will be shifted by each shift in note_shifts
        for shift in note_shifts:
            _shift = shift.item() #assume shift is a numpy ndarray
            
            #initialize the note shifted sequence as a list
            note_shifted_seq = [] 
            
            if _shift == 0:
                note_shifted_seq = seq
            else:
                #iterate over each elem of seq, shift it and append to note_shifted seq
                for idx in seq:
                    _idx = idx + _shift #shift the index
                    
                    #if idx is note on, and _idx is also note on, or 
                    #if idx is note_off,and _idx is also note_off, then
                    #add _idx to note_shifted_sequence, else add idx
                    if (0 < idx <= noe and 0 < _idx <= noe) or (noe < idx <= ne and noe < _idx <= ne):
                        note_shifted_seq.append(_idx)
                    else:
                        note_shifted_seq.append(idx)
            #note_shifted_seq = tf.convert_to_tensor(note_shifted_seq) #convert to tensor
            note_shifted_data.append(np.array(note_shifted_seq, dtype=np.int32))
            
            count += 1
            if not sample_random_time:
                print(f'Augmented data sample {count} created')
            else:
                print(f'Note shifted sample {count} created')
    
    #now iterate over the note shifted data to stretch it in time
    time_shifted_data = [] #initialize the set of time_stretched sequences
    if sample_random_time: count = 0
    
    for seq in note_shifted_data:
        # data will be stretched in time by each time_stretch
        # or by random time stretch if sample_random_time
        if sample_random_time:
            time_stretches_ = random.sample(ts, sample_size)
            for _ in time_stretches_:
                time_shifted_seq = stretch_time(seq, _)
                time_shifted_data.append(time_shifted_seq)
                count += 1
                print(f"Augmented data sample {count} created")
            continue
        
        for time_stretch in ts:
            time_shifted_seq = stretch_time(seq, time_stretch)
            time_shifted_data.append(time_shifted_seq)
            if time_stretch != 1:
                count += 1
                print(f"Augmented data sample {count} created")

    #output the data
    return time_shifted_data

"""TEST MODEL ACCURACY"""

def generate_scale(note=60, delta_times=[500], velocities=list(np.arange(9, 24)), 
                   mode='ionian', octaves=1):
    """
    generates a scale based on the input note and mode
    """
    note = note + 1
    iter_times = iter([time_cutter(dt) for dt in delta_times])
    for i, velocity in enumerate(velocities):
        if velocity > velocity_events:
            velocities[i] = velocity_to_bin(velocity)
    iter_vel = iter(velocities)
    
    modes = ['ionian', 'dorian', 'phrygian', 'lydian', 'mixolydian', 'aeolian',
             'locrian', 'major', 'harmonic', 'melodic']

    mode_steps = np.array([[0, 2, 4, 5, 7, 9, 11, 12, 11, 9, 7, 5, 4, 2, 0],
                           [0, 2, 3, 5, 7, 9, 10, 12, 10, 9, 7, 5, 3, 2, 0],
                           [0, 1, 3, 5, 7, 8, 10, 12, 10, 8, 7, 5, 3, 1, 0],
                           [0, 2, 4, 6, 7, 9, 11, 12, 11, 9, 7, 6, 4, 2, 0],
                           [0, 2, 4, 5, 7, 9, 10, 12, 10, 9, 7, 5, 4, 2, 0],
                           [0, 2, 3, 5, 7, 8, 10, 12, 10, 8, 7, 5, 3, 2, 0],
                           [0, 1, 3, 5, 6, 8, 10, 12, 10, 8, 6, 5, 3, 1, 0],
                           [0, 2, 4, 5, 7, 9, 11, 12, 11, 9, 7, 5, 4, 2, 0],
                           [0, 2, 3, 5, 7, 8, 11, 12, 11, 8, 7, 5, 3, 2, 0],
                           [0, 2, 3, 5, 7, 9, 11, 12, 10, 8, 7, 5, 3, 2, 0]])
    
    mode_steps = mode_steps[modes.index(mode)]
    
    # get octaves
    middle = mode_steps.max() + 12 * (octaves - 1)
    ascend_ = mode_steps[:len(mode_steps) // 2]
    ascend = ascend_[:]
    descend_ = mode_steps[1 + len(mode_steps) // 2:] + 12 * (octaves - 1)
    descend = descend_[:]
    
    for i in range(octaves - 1):
        ascend_ = ascend_ + 12
        ascend = np.concatenate((ascend, ascend_))
        
        descend_ = descend_ - 12
        descend = np.concatenate((descend, descend_))
    
    mode_steps = np.concatenate((ascend, np.expand_dims(middle, 0), descend))
        
        
    scale_ons = np.add(note, mode_steps)
    scale_offs = np.add(scale_ons, note_on_events)\
    
    idx_list = []
    
    for x, y in zip(scale_ons, scale_offs):
        #get velocity
        try:
            velocity = next(iter_vel)
        except StopIteration:
            iter_vel = iter(velocities)
            velocity = next(iter_vel)
        velocity = vocab.index(f"set_velocity_{velocity}")
        
        # get delta time
        try:
            delta_time = next(iter_times)
        except StopIteration:
            iter_times = iter([time_cutter(dt) for dt in delta_times])
            delta_time = next(iter_times)
            
        # append stuff
        idx_list.append(velocity)
        idx_list.append(x)
        for time in delta_time:
            idx_list.append(vocab.index(f"time_shift_{time - 1}"))
        idx_list.append(y)

    return np.array(idx_list, dtype=np.int32)
