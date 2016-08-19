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
