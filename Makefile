# cleanlogs: gets rid of log files
cleanlogs:
	find . -type f -name '*.err' -delete
	find . -type f -name '*.out' -delete

# cleandata: gets rid of all the .wav files
cleanclips:
	find . -type f -name '*.wav' -delete

# cleanembeds: gets rid of all embedding files
cleanembeds:
	find . -type f -name '*.pt' -delete

#cleanpromptdirs: gets rid of empty prompt folders
cleanpromptdirs:
	find . -type d -name 'prompt_*' -empty -delete

#createaudiodir: creates the audio directory and prompt_dirs
createaudiodir:
	mkdir -p data/audio
	N=$$(wc -l < data/prompt.txt); \
	for i in $$(seq 0 $$((N))); do \
		mkdir -p data/audio/prompt_$$i; \
	done
