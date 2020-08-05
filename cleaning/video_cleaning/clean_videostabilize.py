'''
               AAA               lllllll lllllll   iiii                      
              A:::A              l:::::l l:::::l  i::::i                     
             A:::::A             l:::::l l:::::l   iiii                      
            A:::::::A            l:::::l l:::::l                             
           A:::::::::A            l::::l  l::::l iiiiiii     eeeeeeeeeeee    
          A:::::A:::::A           l::::l  l::::l i:::::i   ee::::::::::::ee  
         A:::::A A:::::A          l::::l  l::::l  i::::i  e::::::eeeee:::::ee
        A:::::A   A:::::A         l::::l  l::::l  i::::i e::::::e     e:::::e
       A:::::A     A:::::A        l::::l  l::::l  i::::i e:::::::eeeee::::::e
      A:::::AAAAAAAAA:::::A       l::::l  l::::l  i::::i e:::::::::::::::::e 
     A:::::::::::::::::::::A      l::::l  l::::l  i::::i e::::::eeeeeeeeeee  
    A:::::AAAAAAAAAAAAA:::::A     l::::l  l::::l  i::::i e:::::::e           
   A:::::A             A:::::A   l::::::ll::::::li::::::ie::::::::e          
  A:::::A               A:::::A  l::::::ll::::::li::::::i e::::::::eeeeeeee  
 A:::::A                 A:::::A l::::::ll::::::li::::::i  ee:::::::::::::e  
AAAAAAA                   AAAAAAAlllllllllllllllliiiiiiii    eeeeeeeeeeeeee  


 _____ _                  _                ___  ______ _____     
/  __ \ |                (_)              / _ \ | ___ \_   _|  _ 
| /  \/ | ___  __ _ _ __  _ _ __   __ _  / /_\ \| |_/ / | |   (_)
| |   | |/ _ \/ _` | '_ \| | '_ \ / _` | |  _  ||  __/  | |      
| \__/\ |  __/ (_| | | | | | | | | (_| | | | | || |    _| |_   _ 
 \____/_|\___|\__,_|_| |_|_|_| |_|\__, | \_| |_/\_|    \___/  (_)
                                   __/ |                         
                                  |___/                          
 _   _ _     _            
| | | (_)   | |           
| | | |_  __| | ___  ___  
| | | | |/ _` |/ _ \/ _ \ 
\ \_/ / | (_| |  __/ (_) |
 \___/|_|\__,_|\___|\___/ 

This script takes a folder of videos (as .mp4s) and converts them
to stabilized .mp4 videos. This is useful if the videos are wobbly 
for focusing on video features.

This is enabled if default_video_features = ['clean_videostabilize']
'''
# import required libraries
try:
    from vidgear.gears import VideoGear
except:
    import os
    os.system('pip3 install vidgear==0.1.8')
import numpy as np
import cv2

def clean_videostabilize(videofile):

    # open any valid video stream with stabilization enabled(`stabilize = True`)
    stream_stab = VideoGear(videofile, stabilize = True).start()

    # open same stream without stabilization for comparison
    stream_org = VideoGear(source = "test.mp4").start()

    # loop over
    while True:

        # read stabilized frames
        frame_stab = stream_stab.read()

        # check for stabilized frame if Nonetype
        if frame_stab is None:
            break

        # read un-stabilized frame
        frame_org = stream_org.read()

        # concatenate both frames
        output_frame = np.concatenate((frame_org, frame_stab), axis=1)

        # put text over concatenated frame
        cv2.putText(
            output_frame, "Before", (10, output_frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX,
            0.6, (0, 255, 0), 2,
        )
        cv2.putText(
            output_frame, "After", (output_frame.shape[1] // 2 + 10, output_frame.shape[0] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6, (0, 255, 0), 2,
        )

        # Show output window
        cv2.imshow("Stabilized Frame", output_frame)

        # check for 'q' key if pressed
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    # close output window
    cv2.destroyAllWindows()

    # safely close both video streams
    stream_org.stop()
    stream_stab.stop()
