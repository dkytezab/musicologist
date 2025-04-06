# cleanlogs: gets rid of log files
cleanlogs:
	find . -type f -name '*.err' -delete
	find . -type f -name '*.out' -delete

# cleandata: gets rid of all the .pt files
cleanclips:
	find . -type f -name '*.pt' -delete

# cleanembeds: gets rid of all embedding files
cleanembeds:
	find . -type f -name '*.npy' -delete
