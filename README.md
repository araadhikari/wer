# wer

Using text_sequence_alignment.py file for word error rate (for child speech recognition evaluation):

	1. Make sure to have speech api results and the hand transcription .txt files
	2. Directory for the speech api results (query_dir) and hand transcriptions (ref_dir) files needs to be set.
	3. Each file the speech api results directory is filtered and processed as one big string.
	4. Same for the files in the hand transcriptions directory.
	5. Using swalign python, with modified changes to the __init__.py file. Modified file provided.
	4. Swalign used to get the alignment.match() using the two strings as parameters (querys_line,ref_line).
	5. Aligned_act_filtering and aligned_rec_filtering help with filtering the alignment results for wer use.
	6. WER calculates the word error rate based on # of insertions, substitutions, deletions
	   WER from https://martin-thoma.com/word-error-rate-calculation/

***********************************************

Using text_sequence_alignment_matches for the phrase matches (exact and similar):

	  1. Got Cyber4_Sheet_storyab.xlsx file by only keeping the conditions sheet in Cyber4_Sheet.xlsx.
	  2. Change directory for the child storytelling files in the code's main function to the cyber4storytellingchild folder dir.
	  3. Have the cyber4_robot_story_A.txt and cyber4_robot_story_B.txt files together with the text_sequence_alignment_matches.py file.
	  4. Results will be printed on the console and matches.txt file.
	  5. For exact matches, default number of words in a phrase is 3, 2 for similar matches.
	     LongestSubstringFinder code from http://stackoverflow.com/questions/18715688/find-common-substring-between-two-strings
