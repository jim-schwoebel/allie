import numpy as np
import peakutils as pu

def get_F_0( signal, rate, time_step = 0.0, min_pitch = 75, max_pitch = 600, 
            max_num_cands = 15, silence_thres = .03, voicing_thres = .45, 
            octave_cost = .01, octave_jump_cost = .35,
            voiced_unvoiced_cost = .14, accurate = False, pulse = False ):
    """
    Computes median Fundamental Frequency ( :math:`F_0` ).
    The fundamental frequency ( :math:`F_0` ) of a signal is the lowest 
    frequency, or the longest wavelength of a periodic waveform. In the context
    of this algorithm, :math:`F_0` is calculated by segmenting a signal into 
    frames, then for each frame the most likely candidate is chosen from the 
    lowest possible frequencies to be :math:`F_0`. From all of these values, 
    the median value is returned. More specifically, the algorithm filters out 
    frequencies higher than the Nyquist Frequency from the signal, then 
    segments the signal into frames of at least 3 periods of the minimum 
    pitch. For each frame, it then calculates the normalized autocorrelation 
    ( :math:`r_a` ), or the correlation of the signal to a delayed copy of 
    itself. :math:`r_a` is calculated according to Boersma's paper 
    ( referenced below ), which is an improvement of previous methods. 
    :math:`r_a` is estimated by dividing the autocorrelation of the windowed 
    signal by the autocorrelation of the window. After :math:`r_a` is 
    calculated the maxima values of :math:`r_a` are found. These points
    correspond to the lag domain, or points in the delayed signal, where the 
    correlation value has peaked. The higher peaks indicate a stronger 
    correlation. These points in the lag domain suggest places of wave 
    repetition and are the candidates for :math:`F_0`. The best candidate for 
    :math:`F_0` of each frame is picked by a cost function, a function that 
    compares the cost of transitioning from the best :math:`F_0` of the 
    previous frame to all possible :math:`F_0's` of the current frame. Once the
    path of :math:`F_0's` of least cost has been determined, the median 
    :math:`F_0` of all voiced frames is returned.
    This algorithm is adapted from: 
    http://www.fon.hum.uva.nl/david/ba_shs/2010/Boersma_Proceedings_1993.pdf
    and from:
    https://github.com/praat/praat/blob/master/fon/Sound_to_Pitch.cpp
    
    .. note::
        It has been shown that depressed and suicidal men speak with a reduced 
        fundamental frequency range, ( described in: 
        http://ameriquests.org/index.php/vurj/article/download/2783/1181 ) and 
        patients responding well to depression treatment show an increase in 
        their fundamental frequency variability ( described in :
        https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3022333/ ). Because 
        acoustical properties of speech are the earliest and most consistent 
        indicators of mood disorders, early detection of fundamental frequency 
        changes could significantly improve recovery time for disorders with
        psychomotor symptoms.
        
    Args:
        signal ( numpy.ndarray ): This is the signal :math:`F_0` will be calculated from.
        rate ( int ): This is the number of samples taken per second.
        time_step ( float ): ( optional, default value: 0.0 ) The measurement, in seconds, of time passing between each frame. The smaller the time_step, the more overlap that will occur. If 0 is supplied the degree of oversampling will be equal to four.
        min_pitch ( float ): ( optional, default value: 75 ) This is the minimum value to be returned as pitch, which cannot be less than or equal to zero.
        max_pitch ( float ): ( optional, default value: 600 ) This is the maximum value to be returned as pitch, which cannot be greater than the Nyquist Frequency.
        max_num_cands ( int ): ( optional, default value: 15 ) This is the maximum number of candidates to be considered for each frame, the unvoiced candidate ( i.e. :math:`F_0` equal to zero ) is always considered.
        silence_thres ( float ): ( optional, default value: 0.03 ) Frames that do not contain amplitudes above this threshold ( relative to the global maximum amplitude ), are probably silent.
        voicing_thres ( float ): ( optional, default value: 0.45 ) This is the strength of the unvoiced candidate, relative to the maximum possible :math:`r_a`. To increase the number of unvoiced decisions, increase this value.
        octave_cost ( float ): ( optional, default value: 0.01 per octave ) This is the degree of favouring of high-frequency candidates, relative to the maximum possible :math:`r_a`. This is necessary because in the case of a perfectly periodic signal, all undertones of :math:`F_0` are equally strong candidates as :math:`F_0` itself. To more strongly favour recruitment of high-frequency candidates, increase this value.
        octave_jump_cost ( float ): ( optional, default value: 0.35 ) This is degree of disfavouring of pitch changes, relative to the maximum possible :math:`r_a`. To decrease the number of large frequency jumps, increase this value. 
        voiced_unvoiced_cost ( float ): ( optional, default value: 0.14 ) This is the degree of disfavouring of voiced/unvoiced transitions, relative to the maximum possible :math:`r_a`. To decrease the number of voiced/unvoiced transitions, increase this value.
        accurate ( bool ): ( optional, default value: False ) If False, the window is a Hanning window with a length of :math:`\\frac{ 3.0} {min\_pitch}`. If True, the window is a Gaussian window with a length of :math:`\\frac{6.0}{min\_pitch}`, i.e. twice the length.
        pulse ( bool ): ( optional, default value: False ) If False, the function returns a list containing only the median :math:`F_0`. If True, the function returns a list with all values necessary to calculate pulses. This list contains the median :math:`F_0`, the frequencies for each frame in a list, a list of tuples containing the beginning time of the frame, and the ending time of the frame, and the signal filtered by the Nyquist Frequency. The indicies in the second and third list correspond to each other.
        
    Returns:
        list: Index 0 contains the median :math:`F_0` in hz. If pulse is set 
        equal to True, indicies 1, 2, and 3 will contain: a list of all voiced 
        periods in order, a list of tuples of the beginning and ending time
        of a voiced interval, with each index in the list corresponding to the 
        previous list, and a numpy.ndarray of the signal filtered by the 
        Nyquist Frequency. If pulse is set equal to False, or left to the 
        default value, then the list will only contain the median :math:`F_0`.

    Raises:
        ValueError: min_pitch has to be greater than zero.
        ValueError: octave_cost isn't in [ 0, 1 ].
        ValueError: silence_thres isn't in [ 0, 1 ].
        ValueError: voicing_thres isn't in [ 0, 1 ].
        ValueError: max_pitch can't be larger than Nyquist Frequency.

    Example:
        The example below demonstrates what different outputs this function 
        gives, using a synthesized signal.
        
        >>> import numpy as np
        >>> from matplotlib import pyplot as plt
        >>> domain = np.linspace( 0, 6, 300000 )
        >>> rate = 50000
        >>> y = lambda x: np.sin( 2 * np.pi * 140 * x )
        >>> signal = y( domain )
        >>> get_F_0( signal, rate )
        [ 139.70588235294116 ]
        
        >>> get_F_0( signal, rate, voicing_threshold = .99, accurate = True )
        [ 139.70588235294116 ]
        
        >>> w, x, y, z = get_F_0( signal, rate, pulse = True )
        >>> print( w )
        139.70588235294116
        
        >>> print( x[ :5 ] )
        [ 0.00715789  0.00715789  0.00715789  0.00715789  0.00715789 ]
        
        >>> print( y[ :5 ] )
        [ ( 0.002500008333361111, 0.037500125000416669 ),
        ( 0.012500041666805555, 0.047500158333861113 ),
        ( 0.022500075000249999, 0.057500191667305557 ),
        ( 0.032500108333694447, 0.067500225000749994 ),
        ( 0.042500141667138891, 0.077500258334194452 ) ]
        
        >>> print( z[ : 5 ] )
        [ 0.          0.01759207  0.0351787   0.05275443  0.07031384 ]
        
        The example below demonstrates the algorithms ability to adjust for 
        signals with dynamic frequencies, by comparing a plot of a synthesized 
        signal with an increasing frequency, and the calculated frequencies for 
        that signal.
        
        >>> domain = np.linspace( 1, 2, 10000 )
        >>> rate = 10000
        >>> y = lambda x : np.sin( x ** 8 )
        >>> signal = y( domain )
        >>> median_F_0, periods, time_vals, modified_sig = get_F_0( signal, 
        rate, pulse = True )
        >>> plt.subplot( 211 )
        >>> plt.plot( domain, signal )
        >>> plt.title( "Synthesized Signal" )
        >>> plt.ylabel( "Amplitude" )
        >>> plt.subplot( 212 )
        >>> plt.plot( np.linspace( 1, 2, len( periods ) ), 1.0 / np.array( 
        periods ) )
        >>> plt.title( "Frequencies of Signal" )
        >>> plt.xlabel( "Samples" )
        >>> plt.ylabel( "Frequency" )
        >>> plt.suptitle( "Comparison of Synthesized Signal and it's Calculated Frequencies" )
        >>> plt.show()
        
    .. figure::  figures/F_0_synthesized_sig.png
       :align:   center
    """
    if min_pitch <= 0:
        raise ValueError( "min_pitch has to be greater than zero." )
        
    if max_num_cands < max_pitch / min_pitch:
        max_num_cands = int( max_pitch / min_pitch )
    
    initial_len = len( signal )
    total_time = initial_len / float( rate )
    tot_time_arr = np.linspace( 0, total_time, initial_len )
    max_place_poss  = 1.0 / min_pitch
    min_place_poss  = 1.0 / max_pitch
    #to silence formants
    min_place_poss2 = 0.5 / max_pitch
    
    if accurate: pds_per_window = 6.0
    else:        pds_per_window = 3.0
    
    #degree of oversampling is 4    
    if time_step <= 0: time_step = ( pds_per_window / 4.0 ) / min_pitch
                                   
    w_len = pds_per_window / min_pitch
    #correcting for time_step       
    octave_jump_cost     *= .01 / time_step
    voiced_unvoiced_cost *= .01 / time_step 
    
    Nyquist_Frequency = rate /  2.0
    upper_bound = .95 * Nyquist_Frequency
    zeros_pad = 2 ** ( int( np.log2( initial_len ) ) + 1 ) - initial_len
    signal = np.hstack( ( signal, np.zeros( zeros_pad ) ) )
    fft_signal = np.fft.fft( signal )
    fft_signal[ int( upper_bound ) : -int( upper_bound ) ] = 0
    sig = np.fft.ifft( fft_signal )
    sig = sig[ :initial_len ].real
             
    #checking to make sure values are valid
    if Nyquist_Frequency < max_pitch:
        raise ValueError( "max_pitch can't be larger than Nyquist Frequency." )
    if octave_cost < 0 or octave_cost > 1:
        raise ValueError( "octave_cost isn't in [ 0, 1 ]" )            
    if voicing_thres< 0 or voicing_thres > 1:
        raise ValueError( "voicing_thres isn't in [ 0, 1 ]" ) 
    if silence_thres < 0 or silence_thres > 1:
        raise ValueError( "silence_thres isn't in [ 0, 1 ]" )
        
    #finding number of samples per frame and time_step
    frame_len = int( w_len * rate + .5 )
    time_len  = int( time_step  * rate + .5 )
        
    #initializing list of candidates for F_0, and their strengths
    best_cands, strengths, time_vals = [], [], []
    
    #finding the global peak the way Praat does
    global_peak = max( abs( sig - sig.mean() ) ) 
    print(type(global_peak),'\n')
    e = np.e
    inf = np.inf
    log = np.log2
    start_i = 0
    while start_i < len( sig ) - frame_len :
        end_i = start_i + frame_len
        segment = sig[ start_i : end_i ]
        
        if accurate:
            t = np.linspace( 0, w_len, len( segment ) )
            numerator = e ** ( -12.0 * ( t / w_len - .5 ) ** 2.0 ) - e ** -12.0
            denominator = 1.0 - e ** -12.0
            window = numerator / denominator
            interpolation_depth = 0.25
        else: 
            window = np.hanning( len( segment ) )    
            interpolation_depth = 0.50
        
        #shave off ends of time intervals to account for overlapping
        start_time = tot_time_arr[ start_i + int( time_len / 4.0 ) ]
        stop_time  = tot_time_arr[ end_i   - int( time_len / 4.0 ) ]
        time_vals.append( ( start_time, stop_time ) )
          
        start_i += time_len
        
        long_pd_i = int( rate / min_pitch )
        half_pd_i = int( long_pd_i / 2.0 + 1 )
        
        long_pd_cushion = segment[ half_pd_i : - half_pd_i ]  
        #finding local peak and local mean the way Praat does
        #local mean is found by looking a longest period to either side of the 
        #center of the frame, and using only the values within this interval to 
        #calculate the local mean, and similarly local peak is found by looking
        #a half of the longest period to either side of the center of the 
        #frame, ( after the frame has windowed ) and choosing the absolute 
        #maximum in this interval
        local_mean = long_pd_cushion.mean() 
        segment = segment - local_mean
        segment *= window
        half_pd_cushion = segment[ long_pd_i : -long_pd_i ] 
        local_peak = max( abs( half_pd_cushion ) )
        if local_peak == 0:
            #shortcut -> complete silence and only candidate is silent candidate
            best_cands.append( [ inf ] )
            strengths.append( [ voicing_thres + 2 ] )
        else:
            #calculating autocorrelation, based off steps 3.2-3.10
            intensity = local_peak / float( global_peak )
         
            N = len( segment )
            nFFT = 2 ** int( log( ( 1.0 + interpolation_depth ) * N ) + 1 )
            window  = np.hstack( (   window, np.zeros( nFFT - N ) ) ) 
            segment = np.hstack( (  segment, np.zeros( nFFT - N ) ) )
            x_fft = np.fft.fft( segment )
            r_a = np.real( np.fft.fft( x_fft * np.conjugate( x_fft ) ) )
            r_a = r_a[ : int( N / pds_per_window ) ]
                       
            x_fft = np.fft.fft( window )
            r_w = np.real( np.fft.fft( x_fft * np.conjugate( x_fft ) ) )
            r_w = r_w[ : int( N / pds_per_window ) ]
            r_x = r_a / r_w
            r_x /= r_x[ 0 ]
            
            #creating an array of the points in time corresponding to sampled 
            #autocorrelation of the signal ( r_x )
            time_array = np.linspace( 0 , w_len / pds_per_window, len( r_x ) )
            peaks = pu.indexes( r_x , thres = 0 )
            max_values, max_places = r_x[ peaks ], time_array[ peaks ]
            
            #only consider places that are voiced over a certain threshold
            max_places = max_places[ max_values > 0.5 * voicing_thres ]
            max_values = max_values[ max_values > 0.5 * voicing_thres ]  
            
            for i in range( len( max_values ) ):
                #reflecting values > 1 through 1.
                if max_values[ i ] > 1.0 : 
                    max_values[ i ] = 1.0 / max_values[ i ]
            
            #calculating the relative strength value
            rel_val = [ val - octave_cost * log( place * min_pitch ) for 
                        val, place in zip( max_values, max_places ) ]
            
            if len( max_values ) > 0.0 :
                #finding the max_num_cands-1 maximizers, and maximums, then 
                #calculating their strengths ( eq. 23 and 24 ) and accounting for 
                #silent candidate
                max_places = [ max_places[ i ] for i in np.argsort( rel_val )[
                        -max_num_cands + 1 : ] ] 
                max_values = [ max_values[ i ] for i in np.argsort( rel_val )[
                        -max_num_cands + 1 : ] ] 
                max_places = np.array( max_places )
                max_values = np.array( max_values )
                
                rel_val = list(np.sort( rel_val )[ -max_num_cands + 1 : ] )
                #adding the silent candidate's strength to strengths
                rel_val.append( voicing_thres + max( 0, 2 - ( intensity / 
                        ( silence_thres / ( 1 + voicing_thres ) ) ) ) )
                
                #inf is our silent candidate
                max_places = np.hstack( ( max_places, inf ) )
                
                best_cands.append( list( max_places ) )
                strengths.append( rel_val )
            else:
                #if there are no available maximums, only account for silent 
                #candidate
                best_cands.append( [ inf ] )
                strengths.append( [ voicing_thres + max( 0, 2 - intensity /
                        ( silence_thres / ( 1 + voicing_thres ) ) ) ] )
            
    #Calculates smallest costing path through list of candidates ( forwards ), 
    #and returns path.
    best_total_cost, best_total_path = -inf, []
    #for each initial candidate find the path of least cost, then of those 
    #paths, choose the one with the least cost.
    for cand in range( len( best_cands[ 0 ] ) ):
        start_val = best_cands[ 0 ][ cand ]
        total_path = [ start_val ]
        level = 1
        prev_delta = strengths[ 0 ][ cand ]
        maximum = -inf
        while level < len( best_cands ) :
            prev_val = total_path[ -1 ]
            best_val  = inf
            for j in range( len( best_cands[ level ] ) ):
                cur_val   = best_cands[ level ][ j ] 
                cur_delta =  strengths[ level ][ j ]
                cost = 0
                cur_unvoiced  = cur_val  == inf or cur_val  < min_place_poss2
                prev_unvoiced = prev_val == inf or prev_val < min_place_poss2 
                
                if cur_unvoiced:
                    #both voiceless
                    if prev_unvoiced: 
                        cost = 0 
                    #voiced-to-unvoiced transition
                    else:             
                        cost = voiced_unvoiced_cost 
                else:
                    #unvoiced-to-voiced transition
                    if prev_unvoiced: 
                        cost = voiced_unvoiced_cost
                    #both are voiced
                    else:             
                        cost = octave_jump_cost * abs( log( cur_val /
                                                           prev_val ) )
                            
                #The cost for any given candidate is given by the transition 
                #cost, minus the strength of the given candidate
                value = prev_delta - cost + cur_delta
                if value > maximum: maximum, best_val = value, cur_val
                    
            prev_delta = maximum        
            total_path.append( best_val )
            level += 1
            
        if maximum > best_total_cost: 
            best_total_cost, best_total_path = maximum, total_path
            
    f_0_forth = np.array( best_total_path )        
    
    #Calculates smallest costing path through list of candidates ( backwards ), 
    #and returns path. Going through the path backwards introduces frequencies 
    #previously marked as unvoiced, or increases undertones, to decrease 
    #frequency jumps
    
    best_total_cost, best_total_path2 = -inf, []
    
    #Starting at the end, for each initial candidate find the path of least 
    #cost, then of those paths, choose the one with the least cost.
    for cand in range( len( best_cands[ -1 ] ) ):
        start_val = best_cands[ -1 ][ cand ]
        total_path = [ start_val ]
        level = len( best_cands ) - 2
        prev_delta = strengths[ -1 ][ cand ]
        maximum = -inf
        while level > -1 :
            prev_val = total_path[ -1 ]
            best_val  = inf
            for j in range( len( best_cands[ level ] ) ):
                cur_val   = best_cands[ level ][ j ] 
                cur_delta =  strengths[ level ][ j ]
                cost = 0
                cur_unvoiced  = cur_val  == inf or cur_val  < min_place_poss2 
                prev_unvoiced = prev_val == inf or prev_val < min_place_poss2 
                
                if cur_unvoiced:
                    #both voiceless
                    if prev_unvoiced: 
                        cost = 0 
                    #voiced-to-unvoiced transition
                    else:             
                        cost = voiced_unvoiced_cost 
                else:
                    #unvoiced-to-voiced transition
                    if prev_unvoiced: 
                        cost = voiced_unvoiced_cost
                    #both are voiced
                    else:             
                        cost = octave_jump_cost * abs( log( cur_val / 
                                                           prev_val ) )
                    
                #The cost for any given candidate is given by the transition 
                #cost, minus the strength of the given candidate
                value = prev_delta - cost + cur_delta
                if value > maximum: maximum, best_val = value, cur_val
                    
            prev_delta = maximum        
            total_path.append( best_val )
            level -= 1
            
        if maximum > best_total_cost: 
            best_total_cost, best_total_path2 = maximum, total_path
            
    f_0_back = np.array( best_total_path2 ) 
    #reversing f_0_backward so the initial value corresponds to first frequency
    f_0_back = f_0_back[ -1 : : -1 ] 

    #choose the maximum frequency from each path for the total path
    f_0 = np.array( [ min( i, j ) for i, j in zip( f_0_forth, f_0_back ) ] )
    
    if pulse:
        #removing all unvoiced time intervals from list
        removed = 0
        for i in range( len( f_0 ) ):
            if f_0[ i ] > max_place_poss or f_0[ i] < min_place_poss:
                time_vals.remove( time_vals[ i - removed ] )
                removed += 1
      
    for i in range( len( f_0 ) ):
        #if f_0 is voiceless assign occurance of peak to inf -> when divided  
        #by one this will give us a frequency of 0, corresponding to a unvoiced
        #frame
        if f_0[ i ] > max_place_poss or f_0[ i ] < min_place_poss :
            f_0[ i ] = inf
              
    f_0 = f_0[ f_0 < inf ]
    if pulse:              
        return [ np.median( 1.0 / f_0 ), list( f_0 ), time_vals, signal ]
    if len( f_0 ) == 0:    
        return [ 0 ]
    else:                   
        return [ np.median( 1.0 / f_0 ) ]   

def get_HNR( signal, rate, time_step = 0, min_pitch = 75, 
             silence_threshold = .1, periods_per_window = 4.5 ):
    """
    Computes mean Harmonics-to-Noise ratio ( HNR ).
    The Harmonics-to-Noise ratio ( HNR ) is the ratio
    of the energy of a periodic signal, to the energy of the noise in the 
    signal, expressed in dB. This value is often used as a measure of 
    hoarseness in a person's voice. By way of illustration, if 99% of the 
    energy of the signal is in the periodic part and 1% of the energy is in 
    noise, then the HNR is :math:`10 \cdot log_{10}( \\frac{99}{1} ) = 20`. 
    A HNR of 0 dB means there is equal energy in harmonics and in noise. The 
    first step for HNR  determination of a signal, in the context of this 
    algorithm, is to set the maximum frequency allowable to the signal's 
    Nyquist  Frequency. Then the signal is segmented into frames of length 
    :math:`\\frac{periods\_per\_window}{min\_pitch}`. Then for each frame, it
    calculates the normalized autocorrelation ( :math:`r_a` ), or the 
    correlation of the signal  to a delayed copy of itself. :math:`r_a` is 
    calculated according to Boersma's paper ( referenced below ). The highest 
    peak is picked from :math:`r_a`. If the height of this peak is larger than 
    the strength of the silent candidate, then the HNR for this frame is 
    calculated from that peak. The height of the peak corresponds to the energy
    of the periodic part of the signal. Once the HNR value has been calculated 
    for all voiced frames, the mean is taken from these values and returned.
    This algorithm is adapted from: 
    http://www.fon.hum.uva.nl/david/ba_shs/2010/Boersma_Proceedings_1993.pdf
    and from:
    https://github.com/praat/praat/blob/master/fon/Sound_to_Harmonicity.cpp
    
    .. note::
        The Harmonics-to-Noise ratio of a person's voice is strongly negatively
        correlated to depression severity ( described in: 
        https://ll.mit.edu/mission/cybersec/publications/publication-files/full_papers/2012_09_09_MalyskaN_Interspeech_FP.pdf )
        and can be used as an early indicator of depression, and suicide risk. 
        After this indicator has been realized, preventative medicine can be 
        implemented, improving recovery time or even preventing further 
        symptoms.
            
    Args:
        signal ( numpy.ndarray ): This is the signal the HNR will be calculated from.
        rate ( int ): This is the number of samples taken per second.
        time_step ( float ): ( optional, default value: 0.0 ) This is the measurement, in seconds, of time passing between each frame. The smaller the time_step, the more overlap that will occur. If 0 is supplied, the degree of oversampling will be equal to four.
        min_pitch ( float ): ( optional, default value: 75 ) This is the minimum value to be returned as pitch, which cannot be less than or equal to zero
        silence_threshold ( float ): ( optional, default value: 0.1 ) Frames that do not contain amplitudes above this threshold ( relative to the global maximum amplitude ), are considered silent.
        periods_per_window ( float ): ( optional, default value: 4.5 ) 4.5 is best for speech. The more periods contained per frame, the more the algorithm becomes sensitive to dynamic changes in the signal.
        
    Returns:
        float: The mean HNR of the signal expressed in dB.
        
    Raises:
        ValueError: min_pitch has to be greater than zero.
        ValueError: silence_threshold isn't in [ 0, 1 ].

    Example:
        The example below adjusts parameters of the function, using the same 
        synthesized signal with added noise, to demonstrate the stability of 
        the function.
        
        >>> import numpy as np
        >>> from matplotlib import pyplot as plt
        >>> domain = np.linspace( 0, 6, 300000 )
        >>> rate = 50000
        >>> y = lambda x:( 1 + .3 * np.sin( 2 * np.pi * 140 * x ) ) * np.sin( 
        2 * np.pi * 140 * x )
        >>> signal = y( domain ) + .2 * np.random.random( 300000 )
        >>> get_HNR( signal, rate )
        21.885338007330802
        
        >>> get_HNR( signal, rate, periods_per_window = 6 )
        21.866307805597849
        
        >>> get_HNR( signal, rate, time_step = .04, periods_per_window = 6 )
        21.878451649148804
        
        We'd expect an increase in noise to reduce HNR and similar energies
        in noise and harmonics to produce a HNR that approaches zero. This is
        demonstrated below.
        
        >>> signals = [ y( domain ) + i / 10.0 * np.random.random( 300000 ) for
        i in range( 1, 11 ) ]
        >>> HNRx10 = [ get_HNR( sig, rate ) for sig in signals ]
        >>> plt.plot( np.linspace( .1, 1, 10 ), HNRx10 )
        >>> plt.xlabel( "Amount of Added Noise" )
        >>> plt.ylabel( "HNR" )
        >>> plt.title( "HNR Values of Signals with Added Noise" )
        >>> plt.show()
        
    .. figure::  figures/HNR_values_added_noise.png
       :align:   center
    """
    #checking to make sure values are valid
    if min_pitch <= 0:
        raise ValueError( "min_pitch has to be greater than zero." )
    if silence_threshold < 0 or silence_threshold > 1:
        raise ValueError( "silence_threshold isn't in [ 0, 1 ]." )
    #degree of overlap is four
    if time_step <= 0: time_step = ( periods_per_window / 4.0 ) / min_pitch 
                                   
    Nyquist_Frequency = rate / 2.0
    max_pitch = Nyquist_Frequency
    global_peak = max( abs( signal - signal.mean() ) ) 
    
    window_len = periods_per_window / float( min_pitch )
    
    #finding number of samples per frame and time_step
    frame_len = int( window_len * rate )
    t_len = int( time_step * rate )
    
    #segmenting signal, there has to be at least one frame
    num_frames = max( 1, int( len( signal ) / t_len + .5 ) ) 
    
    seg_signal = [ signal[ int( i * t_len ) : int( i  * t_len ) + frame_len ]  
                                           for i in range( num_frames + 1 ) ]

    #initializing list of candidates for HNR
    best_cands = []
    for index in range( len( seg_signal ) ):
        
        segment = seg_signal[ index ]
        #ignoring any potential empty segment
        if len( segment) > 0:
            window_len = len( segment ) / float( rate )
    
            #calculating autocorrelation, based off steps 3.2-3.10
            segment = segment - segment.mean()
            local_peak = max( abs( segment ) ) 
            if local_peak == 0 :
                best_cands.append( .5 )
            else:
                intensity = local_peak / global_peak 
                window = np.hanning( len( segment ) )
                segment *= window
               
                N = len( segment )
                nsampFFT = 2 ** int( np.log2( N ) + 1 )
                window  = np.hstack( (   window, np.zeros( nsampFFT - N ) ) ) 
                segment = np.hstack( (  segment, np.zeros( nsampFFT - N ) ) )
                x_fft = np.fft.fft( segment )
                r_a = np.real( np.fft.fft( x_fft * np.conjugate( x_fft ) ) )
                r_a = r_a[ : N ]
                r_a = np.nan_to_num( r_a )
                
                x_fft = np.fft.fft( window )
                r_w = np.real( np.fft.fft( x_fft * np.conjugate( x_fft ) ) )
                r_w = r_w[ : N ]
                r_w = np.nan_to_num( r_w )
                r_x = r_a / r_w
                
                r_x /= r_x[ 0 ]
                #creating an array of the points in time corresponding to the 
                #sampled autocorrelation of the signal ( r_x )
                time_array = np.linspace( 0, window_len, len( r_x ) )
                i = pu.indexes( r_x )
                max_values, max_places = r_x[ i ], time_array[ i ]
                max_place_poss = 1.0 / min_pitch
                min_place_poss = 1.0 / max_pitch
        
                max_values = max_values[ max_places >= min_place_poss ]
                max_places = max_places[ max_places >= min_place_poss ]
                
                max_values = max_values[ max_places <= max_place_poss ]
                max_places = max_places[ max_places <= max_place_poss ]
                
                for i in range( len( max_values ) ):
                    #reflecting values > 1 through 1.
                    if max_values[ i ] > 1.0 : 
                        max_values[ i ] = 1.0 / max_values[ i ]
                
                #eq. 23 and 24 with octave_cost, and voicing_threshold set to zero
                if len( max_values ) > 0:
                    strengths = [ max( max_values ), max( 0, 2 - ( intensity /
                                                            ( silence_threshold ) ) ) ]
                #if the maximum strength is the unvoiced candidate, then .5 
                #corresponds to HNR of 0
                    if np.argmax( strengths ):
                        best_cands.append( 0.5 )  
                    else:
                        best_cands.append( strengths[ 0 ] )
                else:
                    best_cands.append( 0.5 )
    
    best_cands = np.array( best_cands )
    best_cands = best_cands[ best_cands > 0.5 ]
    if len(best_cands) == 0:
        return 0
    #eq. 4
    best_cands = 10.0 * np.log10( best_cands / ( 1.0 - best_cands ) )
    best_candidate = np.mean( best_cands )
    return best_candidate
    
def get_Pulses( signal, rate, min_pitch = 75, max_pitch = 600,
                include_max = False, include_min = True ):
    """
    Computes glottal pulses of a signal.
    This algorithm relies on the voiced/unvoiced decisions and fundamental 
    frequencies, calculated for each voiced frame by get_F_0. For every voiced 
    interval, a list of points is created by finding the initial point 
    :math:`t_1`, which is the absolute extremum ( or the maximum/minimum, 
    depending on your include_max and include_min parameters ) of the amplitude 
    of the sound in the interval 
    :math:`[\ t_{mid} - \\frac{T_0}{2},\ t_{mid} + \\frac{T_0}{2}\ ]`, where 
    :math:`t_{mid}` is the midpoint of the interval, and :math:`T_0` is the 
    period at :math:`t_{mid}`, as can be linearly interpolated from the periods
    acquired from get_F_0. From this point, the algorithm searches for points 
    :math:`t_i` to the left until we reach the left edge of the interval. These
    points are the absolute extrema ( or the maxima/minima ) in the interval
    :math:`[\ t_{i-1} - 1.25 \cdot T_{i-1},\ t_{i-1} - 0.8 \cdot T_{i-1}\ ]`, 
    with :math:`t_{i-1}` being the last found point, and :math:`T_{i-1}` the 
    period at this point. The same is done to the right of :math:`t_1`. The 
    points are returned in consecutive order.
    This algorithm is adapted from: 
    https://pdfs.semanticscholar.org/16d5/980ba1cf168d5782379692517250e80f0082.pdf
    and from:
    https://github.com/praat/praat/blob/master/fon/Sound_to_PointProcess.cpp
     
    .. note::
        This algorithm is a helper function for the jitter algorithm, that 
        returns a list of points in the time domain corresponding to minima or 
        maxima of the signal. These minima or maxima are the sequence of 
        glottal closures in vocal-fold vibration. The distance between 
        consecutive pulses is defined as the wavelength of the signal at this 
        interval, which can be used to later calculate jitter. 
         
    Args:
        signal ( numpy.ndarray ): This is the signal the glottal pulses will be calculated from.
        rate ( int ): This is the number of samples taken per second.
        min_pitch ( float ): ( optional, default value: 75 ) This is the minimum value to be returned as pitch, which cannot be less than or equal to zero
        max_pitch ( float ): ( optional, default value: 600 ) This is the maximum value to be returned as pitch, which cannot be greater than Nyquist Frequency   
        include_max ( bool ): ( optional, default value: False ) This determines if maxima values will be used when calculating pulses
        include_min ( bool ): ( optional, default value: True ) This determines if minima values will be used when calculating pulses
        
    Returns:
        numpy.ndarray: This is an array of points in a time series that
        correspond to the signal's periodicity.
    
    Raises:
        ValueError: include_min and include_max can't both be False
        
    Example:
        Pulses are calculated for a synthesized signal, and the variation in 
        time between each pulse is shown.
        
        >>> import numpy as np
        >>> from matplotlib import pyplot as plt
        >>> domain = np.linspace( 0, 6, 300000 )
        >>> y = lambda x:( 1 + .3 * np.sin( 2 * np.pi * 140 * x ) ) * np.sin( 
        2 * np.pi * 140 * x )
        >>> signal = y( domain ) + .2 * np.random.random( 300000 )
        >>> rate = 50000
        >>> p = get_Pulses( signal, rate )
        >>> print( p[ :5 ] )
        [ 0.00542001  0.01236002  0.01946004  0.02702005  0.03402006 ]
        
        >>> print( np.diff( p[ :6 ] ) )
        [ 0.00694001  0.00710001  0.00756001  0.00700001  0.00712001 ]
        
        >>> p = get_Pulses( signal, rate, include_max = True )
        >>> print( p[ :5 ] )
        [ 0.00886002  0.01608003  0.02340004  0.03038006  0.03732007 ]
        
        >>> print( np.diff( p[ :6 ] ) )
        [ 0.00722001  0.00732001  0.00698001  0.00694001  0.00734001 ]
 
        A synthesized signal, with an increasing frequency, and the calculated 
        pulses of that signal are plotted together to demonstrate the 
        algorithms ability to adapt to dynamic pulses.
        
        >>> domain = np.linspace( 1.85, 2.05, 10000 )
        >>> rate = 50000
        >>> y = lambda x : np.sin( x ** 8 )
        >>> signal = np.hstack( ( np.zeros( 2500 ), y( domain[ 2500: -2500 ] ),
        np.zeros( 2500 ) ) )
        >>> pulses = get_Pulses( signal, rate )
        >>> plt.plot( domain, signal, 'r', alpha = .5, label = "Signal" )
        >>> plt.plot( ( 1.85 + pulses[ 0 ] ) * np.ones ( 5 ), 
        np.linspace( -1, 1, 5 ), 'b', alpha = .5, label = "Pulses" )
        >>> plt.legend()
        >>> for pulse in pulses[ 1: ]:
        >>>     plt.plot( ( 1.85 + pulse ) * np.ones ( 5 ), 
        np.linspace( -1, 1, 5 ), 'b', alpha = .5 )
        >>> plt.xlabel( "Samples" )
        >>> plt.ylabel( "Amplitude" )
        >>> plt.title( "Signal with Pulses, Calculated from Minima of Signal" )
        >>> plt.show() 
        
    .. figure::  figures/Pulses_sig.png
       :align:   center           
    """
    #first calculate F_0 estimates for each voiced interval
    add = np.hstack
    if not include_max and not include_min:
        raise ValueError( "include_min and include_max can't both be False" )
        
    median, period, intervals, signal = get_F_0( signal, rate, 
                                                min_pitch = min_pitch, 
                                                max_pitch = max_pitch, 
                                                pulse = True )
    global_peak = max( abs( signal - signal.mean() ) )
    #points will be a list of points where pulses occur, voiced_intervals will
    #be a list of tuples consisting of voiced intervals with overlap
    #eliminated
    points, voiced_intervals = [], []
    #f_times will be an array of times corresponding to our given frequencies,
    #to be used for interpolating, v_time be an array consisting of all the 
    #points in time that are voiced
    f_times, v_time = np.array( [] ), np.array( [] )
    total_time = np.linspace( 0, len( signal ) / float( rate ), len( signal ) )
    
    for interval in intervals:
        start, stop = interval
        #finding all midpoints for each interval 
        f_times = add( ( f_times, ( start + stop ) / 2.0 ) )
        
    i = 0  
    while i < len( intervals ) - 1 :
        start, stop = intervals[ i ]
        i_start, prev_stop = intervals[ i ]
        #while there is overlap, look to the next interval
        while start <= prev_stop and i < len( intervals ) - 1 :
            prev_start, prev_stop = intervals[ i ]
            i += 1
            start, stop = intervals[ i ]
        if i == len( intervals ) - 1:
            samp = int ( (      stop - i_start ) * rate )
            v_time = add( ( v_time, np.linspace( i_start,      stop, samp ) ) )
            voiced_intervals.append( ( i_start, stop ) )
        else:
            samp = int ( ( prev_stop - i_start ) * rate )
            v_time = add( ( v_time, np.linspace( i_start, prev_stop, samp ) ) )	
            voiced_intervals.append( ( i_start, prev_stop ) )
    
    #interpolate the periods so that each voiced point has a corresponding
    #period attached to it
    periods_interp = np.interp( v_time, f_times, period )

    
    for interval in voiced_intervals:
        start, stop = interval
        midpoint = ( start + stop ) / 2.0
        #out of all the voiced points, look for index of the one that is 
        #closest to our calculated midpoint
        midpoint_index = np.argmin( abs( v_time - midpoint ) )
        midpoint = v_time[ midpoint_index ]
        T_0 = periods_interp[ midpoint_index ]
        frame_start = midpoint - T_0
        frame_stop  = midpoint + T_0
        #finding points, start by looking to the left of the center of the 
        #voiced interval
        while frame_start > start :
            #out of all given time points in signal, find index of closest to
            #start and stop
            frame_start_index = np.argmin( abs( total_time - frame_start ) )
            frame_stop_index  = np.argmin( abs( total_time - frame_stop  ) )
            
            frame = signal[ frame_start_index : frame_stop_index ]
            
            if include_max and include_min: 
                p_index = np.argmax( abs( frame ) ) + frame_start_index
            elif include_max:                  
                p_index = np.argmax( frame )        + frame_start_index
            else:                                 
                p_index = np.argmin( frame )        + frame_start_index 
                                                                     
            if abs( signal[ p_index ] ) > .02333 * global_peak: 
                points.append( total_time[ p_index ] )
                
            t = total_time[ p_index ]
            t_index = np.argmin( abs( v_time - t ) )
            T_0 = periods_interp[ t_index ]
            frame_start = t - 1.25 * T_0
            frame_stop  = t - 0.80 * T_0
            
        T_0 = periods_interp[ midpoint_index ]    
        frame_start = midpoint - T_0
        frame_stop  = midpoint + T_0  
        
        #finding points by now looking to the right of the center of the 
        #voiced interval
        while frame_stop < stop :
            #out of all given time points in signal, find index of closest to
            #start and stop
            frame_start_index = np.argmin( abs( total_time - frame_start ) )
            frame_stop_index  = np.argmin( abs( total_time - frame_stop  ) )
            frame = signal[ frame_start_index : frame_stop_index ]
            
            if include_max and include_min: 
                p_index = np.argmax( abs( frame ) ) + frame_start_index
            elif include_max:                  
                p_index = np.argmax( frame )        + frame_start_index
            else:                                 
                p_index = np.argmin( frame )        + frame_start_index 
                                                                     
            if abs( signal[ p_index ] ) > .02333 * global_peak: 
                points.append( total_time[ p_index ] )  
            
            t = total_time[ p_index ]
            t_index = np.argmin( abs( v_time - t ) )
            T_0 = periods_interp[ t_index ]
            frame_start = t + 0.80 * T_0
            frame_stop  = t + 1.25 * T_0 
            
    #returning an ordered array of points with any duplicates removed         
    return np.array( sorted( list( set( points ) ) ) )


def get_Jitter( signal, rate, period_floor = .0001, period_ceiling = .02, 
                max_period_factor = 1.3 ):
    """
    Compute Jitter.
    Jitter is the measurement of random pertubations in period length. For most 
    accurate jitter measurements, calculations are typically performed on long 
    sustained vowels. This algorithm calculates 5 different types of jitter for
    all voiced intervals. Each different type of jitter describes different 
    characteristics of the period pertubations. The 5 types of jitter 
    calculated are absolute jitter, relative jitter, relative average 
    perturbation ( rap ), the 5-point period pertubation quotient ( ppq5 ), and
    the difference of differences of periods ( ddp ).\n
    Absolute jitter is defined as the cycle-to-cycle variation of 
    fundamental frequency, or in other words, the average absolute difference 
    between consecutive periods.
    
    .. math::
        \\frac{1}{N-1}\sum_{i=1}^{N-1}|T_i-T_{i-1}|
        
    Relative jitter is defined as the average absolute difference between 
    consecutive periods ( absolute jitter ), divided by the average period. 
    
    .. math::
        \\frac{\\frac{1}{N-1}\sum_{i=1}^{N-1}|T_i-T_{i-1}|}{\\frac{1}{N}\sum_{i=1}^N T_i}
    
    Relative average perturbation is defined as the average absolute difference
    between a period and the average of it and its two neighbors divided by the
    average period.
    
    .. math::
        \\frac{\\frac{1}{N-1}\sum_{i=1}^{N-1}|T_i-(\\frac{1}{3}\sum_{n=i-1}^{i+1}T_n)|}{\\frac{1}{N}\sum_{i=1}^N T_i}
    
    The 5-point period pertubation quotient is defined as the average absolute 
    difference between a period and the average of it and its 4 closest neighbors 
    divided by the average period.
    
    .. math::
        \\frac{\\frac{1}{N-1}\sum_{i=2}^{N-2}|T_i-(\\frac{1}{5}\sum_{n=i-2}^{i+2}T_n)|}{\\frac{1}{N}\sum_{i=1}^N T_i}
    
    The difference of differences of periods is defined as the relative mean 
    absolute second-order difference of periods, which is equivalent to 3 times
    rap.
    
    .. math::
        \\frac{\\frac{1}{N-2}\sum_{i=2}^{N-1}|(T_{i+1}-T_i)-(T_i-T_{i-1})|}{\\frac{1}{N}\sum_{i=1}^{N}T_i}
    
    After each type of jitter has been calculated the values are 
    returned in a dictionary.
    
    .. warning::
        This algorithm has 4.2% relative error when compared to Praat's values.
        
    This algorithm is adapted from:  
    http://www.lsi.upc.edu/~nlp/papers/far_jit_07.pdf
    and from:
    http://ac.els-cdn.com/S2212017313002788/1-s2.0-S2212017313002788-main.pdf?_tid=0c860a76-7eda-11e7-a827-00000aab0f02&acdnat=1502486243_009951b8dc70e35597f4cd19f8e05930
    and from:
    https://github.com/praat/praat/blob/master/fon/VoiceAnalysis.cpp
    
    .. note::
        Significant differences can occur in jitter and shimmer measurements 
        between different speaking styles, these differences make it possible to 
        use jitter as a feature for speaker recognition ( referenced above ). 
            
    
    Args:
        signal ( numpy.ndarray ): This is the signal the jitter will be calculated from.
        rate ( int ): This is the number of samples taken per second.
        period_floor ( float ): ( optional, default value: .0001 ) This is the shortest possible interval that will be used in the computation of jitter, in seconds. If an interval is shorter than this, it will be ignored in the computation of jitter ( the previous and next intervals will not be regarded as consecutive ).
        period_ceiling ( float ): ( optional, default value: .02 ) This is the longest possible interval that will be used in the computation of jitter, in seconds. If an interval is longer than this, it will be ignored in the computation of jitter ( the previous and next intervals will not be regarded as consecutive ).
        max_period_factor ( float ): ( optional, default value: 1.3 ) This is the largest possible difference between consecutive intervals that will be used in the computation of jitter. If the ratio of the durations of two consecutive intervals is greater than this, this pair of intervals will be ignored in the computation of jitter ( each of the intervals could still take part in the computation of jitter, in a comparison with its neighbor on the other side ).
        
    Returns:
        dict: a dictionary with keys: 'local', 'local, absolute', 'rap', 
        'ppq5', and 'ddp'. The values correspond to each type of jitter.\n
        local jitter is expressed as a ratio of mean absolute period variation 
        to the mean period. \n
        local absolute jitter is given in seconds.\n
        rap is expressed as a ratio of the mean absolute difference between a 
        period and the mean of its 2 neighbors to the mean period.\n
        ppq5 is expressed as a ratio of the mean absolute difference between a 
        period and the mean of its 4 neighbors to the mean period.\n
        ddp is expressed as a ratio of the mean absolute second-order 
        difference to the mean period.
    
    Example:
        In the example below a synthesized signal is used to demonstrate random 
        perturbations in periods, and how get_Jitter responds.
        
        >>> import numpy as np
        >>> domain = np.linspace( 0, 6, 300000 )
        >>> y = lambda x:( 1 - .3 * np.sin( 2 * np.pi * 140 * x ) ) * np.sin( 
        2 * np.pi * 140 * x )
        >>> signal = y( domain ) + .2 * np.random.random( 300000 )
        >>> rate = 50000
        >>> get_Jitter( signal, rate )
        { 'ddp': 0.047411037373434134,
        'local': 0.02581897560637415,
        'local, absolute': 0.00018442618908563846,
        'ppq5': 0.014805010237029443,
        'rap': 0.015803679124478043 } 
        
        >>> get_Jitter( signal, rate, period_floor = .001, 
        period_ceiling = .01, max_period_factor = 1.05 )
        { 'ddp': 0.03264516540374475,
        'local': 0.019927260366800197,
        'local, absolute': 0.00014233584195389132,
        'ppq5': 0.011472274162612033,
        'rap': 0.01088172180124825 }
        
        >>> y = lambda x:( 1 - .3 * np.sin( 2 * np.pi * 140 * x ) ) * np.sin( 
        2 * np.pi * 140 * x )
        >>> signal = y( domain )
        >>> get_Jitter( signal, rate )
        { 'ddp': 0.0015827628114371581, 
        'local': 0.00079043477724730755,
        'local, absolute': 5.6459437833161522e-06,
        'ppq5': 0.00063462518488944565,
        'rap': 0.00052758760381238598 }
    
    """
    pulses = get_Pulses( signal, rate )
    periods = np.diff( pulses )
    
    min_period_factor = 1.0 / max_period_factor
    
    #finding local, absolute
    #described at:
    #http://www.fon.hum.uva.nl/praat/manual/PointProcess__Get_jitter__local__absolute____.html
    sum_total = 0
    num_periods = len( pulses ) - 1
    for i in range( len( periods ) - 1 ):
        p1, p2 = periods[ i ], periods[ i + 1 ]
        
        ratio = p2 / p1
        if (ratio < max_period_factor and ratio > min_period_factor and 
            p1    < period_ceiling    and p1    > period_floor      and
            p2    < period_ceiling    and p2    > period_floor      ):
            
                sum_total += abs( periods[ i + 1 ] - periods[ i ] ) 
        else: num_periods -= 1
                
    absolute = sum_total / ( num_periods - 1 )
    
    #finding local, 
    #described at: 
    #http://www.fon.hum.uva.nl/praat/manual/PointProcess__Get_jitter__local____.html
    sum_total = 0
    num_periods = 0
    
    #duplicating edges so there is no need to test edge cases
    periods = np.hstack( ( periods[ 0 ], periods, periods[ -1 ] ) )
    
    for i in range( len( periods ) - 2):
        p1, p2, p3 = periods[ i ], periods[ i + 1 ], periods[ i + 2 ]
        
        ratio_1, ratio_2 = p1 / p2, p2 / p3
        if (ratio_1 < max_period_factor and ratio_1 > min_period_factor and 
            ratio_2 < max_period_factor and ratio_2 > min_period_factor and 
            p2      < period_ceiling    and p2      > period_floor      ):
            
            sum_total += p2
            num_periods += 1
            
    #removing duplicated edges        
    periods = periods[ 1 : -1 ]
    avg_period = sum_total / ( num_periods ) 
    relative = absolute / avg_period
    
    #finding rap
    #described at: 
    #http://www.fon.hum.uva.nl/praat/manual/PointProcess__Get_jitter__rap____.html
    sum_total = 0
    num_periods = 0
    
    for i in range( len( periods ) - 2 ):
        p1, p2, p3 = periods[ i ], periods[ i + 1 ], periods[ i + 2 ]
        
        ratio_1, ratio_2 = p1 / p2, p2 / p3
        if (ratio_1 < max_period_factor and ratio_1 > min_period_factor and 
            ratio_2 < max_period_factor and ratio_2 > min_period_factor and 
            p1      < period_ceiling    and p1      > period_floor      and
            p2      < period_ceiling    and p2      > period_floor      and
            p3      < period_ceiling    and p3      > period_floor      ):
            
            sum_total += abs( p2 - ( p1 + p2 + p3 ) / 3.0 )
            num_periods += 1
    rap = ( sum_total / num_periods ) / avg_period 
          
    #finding ppq5
    #described at: 
    #http://www.fon.hum.uva.nl/praat/manual/PointProcess__Get_jitter__ppq5____.html
    sum_total = 0
    num_periods = 0
    
    for i in range( len( periods ) - 4 ):
        p1, p2, p3 = periods[ i ], periods[ i + 1 ], periods[ i + 2 ]
        p4, p5 = periods[ i + 3 ], periods[ i + 4 ]
        
        ratio_1, ratio_2, ratio_3, ratio_4 = p1 / p2, p2 / p3, p3 / p4, p4 / p5
        if (ratio_1 < max_period_factor and ratio_1 > min_period_factor and 
            ratio_2 < max_period_factor and ratio_2 > min_period_factor and 
            ratio_3 < max_period_factor and ratio_3 > min_period_factor and 
            ratio_4 < max_period_factor and ratio_4 > min_period_factor and 
            p1      < period_ceiling    and p1      > period_floor      and
            p2      < period_ceiling    and p2      > period_floor      and
            p3      < period_ceiling    and p3      > period_floor      and
            p4      < period_ceiling    and p4      > period_floor      and
            p5      < period_ceiling    and p5      > period_floor      ):
            
            sum_total += abs( p3 - ( p1 + p2 + p3 +p4 + p5 ) / 5.0 )
            num_periods += 1
            
    ppq5 = ( sum_total / num_periods ) / avg_period
            
    #Praat calculates ddp by multiplying rap by 3
    #described at:
    #http://www.fon.hum.uva.nl/praat/manual/PointProcess__Get_jitter__ddp____.html
    
    return {  'local' : relative, 'local, absolute' : absolute, 'rap' : rap, 
                                            'ppq5' : ppq5, 'ddp' : 3 * rap }

