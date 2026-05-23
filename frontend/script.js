document.addEventListener('DOMContentLoaded', () => {
    const dropArea = document.getElementById('drop-area');
    const fileInput = document.getElementById('file-input');
    const selectBtn = document.getElementById('select-btn');
    
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

    // Click events for file selection
    if (dropArea) dropArea.addEventListener('click', () => fileInput.click());
    if (selectBtn) selectBtn.addEventListener('click', (e) => {
        e.stopPropagation();
        fileInput.click();
    });

    // Drag and Drop Logic
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        if (dropArea) dropArea.addEventListener(eventName, preventDefaults, false);
    });

    function preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }

    ['dragenter', 'dragover'].forEach(eventName => {
        if (dropArea) dropArea.addEventListener(eventName, () => dropArea.classList.add('dragover'), false);
    });

    ['dragleave', 'drop'].forEach(eventName => {
        if (dropArea) dropArea.addEventListener(eventName, () => dropArea.classList.remove('dragover'), false);
    });

    if (dropArea) {
        dropArea.addEventListener('drop', (e) => {
            let dt = e.dataTransfer;
            let files = dt.files;
            if (files.length > 0) handleFile(files[0]);
        });
    }

    if (fileInput) {
        fileInput.addEventListener('change', function() {
            if (this.files.length > 0) handleFile(this.files[0]);
        });
    }

    async function handleFile(file) {
        if (!file.type.startsWith('audio/')) {
            alert('Please upload an audio file.');
            return;
        }

        // Generate object URL for original file
        originalAudioUrl = URL.createObjectURL(file);

        // Switch UI to Processing
        if (uploadSection) uploadSection.classList.add('hidden');
        if (processingSection) processingSection.classList.remove('hidden');

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
            if (downloadBtn) downloadBtn.href = enhancedAudioUrl;
            
            // Switch UI to Results
            if (processingSection) processingSection.classList.add('hidden');
            if (resultSection) resultSection.classList.remove('hidden');
            
            // Load enhanced audio by default (toggle is checked)
            if (compareToggle) compareToggle.checked = true;
            updateLabels(true);
            wavesurfer.load(enhancedAudioUrl);
            
        } catch (error) {
            console.error(error);
            alert('Error processing audio. Ensure the backend is running.');
            resetApp();
        }
    }

    // Controls Logic
    if (playPauseBtn) {
        playPauseBtn.addEventListener('click', () => {
            wavesurfer.playPause();
        });
    }

    if (compareToggle) {
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
    }

    function updateLabels(isEnhanced) {
        const labels = document.querySelectorAll('.toggle-container .label');
        if (labels.length >= 2) {
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
    }

    if (resetBtn) {
        resetBtn.addEventListener('click', resetApp);
    }

    function resetApp() {
        if (wavesurfer.isPlaying()) wavesurfer.pause();
        wavesurfer.empty();
        
        if (originalAudioUrl) URL.revokeObjectURL(originalAudioUrl);
        if (enhancedAudioUrl) URL.revokeObjectURL(enhancedAudioUrl);
        
        if (fileInput) fileInput.value = '';
        
        if (resultSection) resultSection.classList.add('hidden');
        if (processingSection) processingSection.classList.add('hidden');
        if (uploadSection) uploadSection.classList.remove('hidden');
    }
});
