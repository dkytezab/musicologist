# cleanlogs: gets rid of log files
cleanlogs:
	find . -type f -name '*.err' -delete
	find . -type f -name '*.out' -delete

# cleandata: gets rid of all the .wav files
cleandata:
	find . -type f -name '*.wav' -delete
