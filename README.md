# wer
For the text alignment:
word error rate, child speech recognition evaluation

for the text_sequence_alignment.py file, need transcripts, groundtruths, and a file with a list of groundtruth file names (filename.txt).
(need these files to be in the same folder as the text_sequence_alignment.py file):

-----
Also, other things needed: numpy, re, difflib, swalign

for swalign, I installed swalign via 'pip install swalign' in cmd line, then made changes to its' __init__.py file under C:\Python35\Lib\site-packages\swalign
the __init__.py file I have is included in here



***********************************************
For the phrase matches:
1. Keep only the conditions sheet in Cyber4_Sheet.xlsx and saved changes as a Cyber4_Sheet_storyab.xlsx file.
2. Make sure to have all the storytelling files in a seperate folder and change directory for it in the code's main function
3. Havethe cyber4_robot_story_A.txt and cyber4_robot_story_B.txt files together with the text_sequence_alignment_matches.py file
4. Results will be printed on the consel and matches.txt file
