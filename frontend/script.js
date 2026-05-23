document.addEventListener('DOMContentLoaded', () => {
    const dropZone = document.getElementById('drop-zone');
    const fileInput = document.getElementById('file-input');
    const uploadSection = document.getElementById('upload-section');
    const processingSection = document.getElementById('processing-section');
    const resultSection = document.getElementById('result-section');
    
    const playPauseBtn = document.getElementById('play-pause-btn');
    const compareToggle = document.getElementById('compare-toggle');
    const downloadBtn = document.getElementById('download-btn');
    const resetBtn = document.getElementById('reset-btn');

    let originalAudioUrl = null;
    let enhancedAudioUrl = null;
    let wavesurfer = null;

    // Initialize Wavesurfer
    wavesurfer = WaveSurfer.create({
        container: '#waveform',
        waveColor: '#94a3b8',
        progressColor: '#8b5cf6',
        cursorColor: '#ec4899',
        barWidth: 3,
        barRadius: 3,
        cursorWidth: 2,
        height: 100,
        barGap: 3
    });

    wavesurfer.on('play', () => playPauseBtn.textContent = '⏸ Pause');
    wavesurfer.on('pause', () => playPauseBtn.textContent = '▶ Play');

    // Drag and Drop Logic
    dropZone.addEventListener('click', () => fileInput.click());

    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        dropZone.addEventListener(eventName, preventDefaults, false);
    });

    function preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }

    ['dragenter', 'dragover'].forEach(eventName => {
        dropZone.addEventListener(eventName, () => dropZone.classList.add('dragover'), false);
    });

    ['dragleave', 'drop'].forEach(eventName => {
        dropZone.addEventListener(eventName, () => dropZone.classList.remove('dragover'), false);
    });

    dropZone.addEventListener('drop', (e) => {
        let dt = e.dataTransfer;
        let files = dt.files;
        if (files.length > 0) handleFile(files[0]);
    });

    fileInput.addEventListener('change', function() {
        if (this.files.length > 0) handleFile(this.files[0]);
    });

    async function handleFile(file) {
        if (!file.type.startsWith('audio/')) {
            alert('Please upload an audio file.');
            return;
        }

        // Generate object URL for original file
        originalAudioUrl = URL.createObjectURL(file);

        // Switch UI to Processing
        uploadSection.classList.add('hidden');
        processingSection.classList.remove('hidden');

        // Send to Backend
        const formData = new FormData();
        formData.append('file', file);

        try {
            const response = await fetch('/api/enhance', {
                method: 'POST',
                body: formData
            });

            if (!response.ok) throw new Error('Enhancement failed.');

            const blob = await response.blob();
            enhancedAudioUrl = URL.createObjectURL(blob);
            
            // Setup Download button
            downloadBtn.href = enhancedAudioUrl;
            
            // Switch UI to Results
            processingSection.classList.add('hidden');
            resultSection.classList.remove('hidden');
            
            // Load enhanced audio by default (toggle is checked)
            compareToggle.checked = true;
            updateLabels(true);
            wavesurfer.load(enhancedAudioUrl);
            
        } catch (error) {
            alert('Error processing audio. Ensure the backend is running.');
            resetApp();
        }
    }

    // Controls Logic
    playPauseBtn.addEventListener('click', () => {
        wavesurfer.playPause();
    });

    compareToggle.addEventListener('change', (e) => {
        const isEnhanced = e.target.checked;
        const currentTime = wavesurfer.getCurrentTime();
        const isPlaying = wavesurfer.isPlaying();
        
        updateLabels(isEnhanced);
        
        // Switch audio source
        wavesurfer.load(isEnhanced ? enhancedAudioUrl : originalAudioUrl).then(() => {
            wavesurfer.seekTo(currentTime / wavesurfer.getDuration());
            if (isPlaying) wavesurfer.play();
        });
    });

    function updateLabels(isEnhanced) {
        const labels = document.querySelectorAll('.toggle-container .label');
        if (isEnhanced) {
            labels[0].classList.remove('active-label');
            labels[1].classList.add('active-label');
            wavesurfer.setOptions({ progressColor: '#8b5cf6' }); // Purple
        } else {
            labels[1].classList.remove('active-label');
            labels[0].classList.add('active-label');
            wavesurfer.setOptions({ progressColor: '#ec4899' }); // Pinkish/Red
        }
    }

    resetBtn.addEventListener('click', resetApp);

    function resetApp() {
        if (wavesurfer.isPlaying()) wavesurfer.pause();
        wavesurfer.empty();
        
        if (originalAudioUrl) URL.revokeObjectURL(originalAudioUrl);
        if (enhancedAudioUrl) URL.revokeObjectURL(enhancedAudioUrl);
        
        fileInput.value = '';
        
        resultSection.classList.add('hidden');
        processingSection.classList.add('hidden');
        uploadSection.classList.remove('hidden');
    }
});
