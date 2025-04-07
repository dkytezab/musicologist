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

# initgen: sets-up the data directory as desired
initgen:
	mkdir -p data/generated
	awk '/^steps:/ {in_steps=1; next} /^[^[:space:]-]/ {in_steps=0} in_steps \
	&& /^\s*-\s*/ {gsub(/- /, "", $$0); print}' diffusion/diff_config.yml | \
	while read step; do \
		mkdir -p data/generated/diff_step_$$step; \
	done

#createaudiodir: creates the audio directory and prompt_dirs
createaudiodir:
	mkdir -p data/audio
	N=$$(wc -l < data/prompt.txt); \
	for i in $$(seq 0 $$((N))); do \
		mkdir -p data/audio/prompt_$$i; \
	done
	
# resetgen: deletes generated and subfolders provided no .wav or .pt files remain. run this after
# running cleanclips and cleanembeds
resetgen:
	find data/generated -type d -name 'diff_step_*' -empty -delete
	find data -type d -name 'generated' -empty -delete
