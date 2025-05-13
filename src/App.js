"use client"

import { useState, useRef, useEffect } from "react"
import { Play, Pause, Volume2, VolumeX, Download, FileText, ArrowLeft, Music } from "lucide-react"

function App() {
  const [inputFile, setInputFile] = useState(null)
  const [separatedAudio, setSeparatedAudio] = useState({
    vocals: null,
    drums: null,
    bass: null,
    other: null,
    notes: { bass: null, drums: null, vocals: null },
    proc_id: null,
    song_name: null
  })
  const [isProcessing, setIsProcessing] = useState(false)
  const [isProcessed, setIsProcessed] = useState(false)
  const [error, setError] = useState(null)
  const [viewingPdf, setViewingPdf] = useState(null)
  const [currentPage, setCurrentPage] = useState("home")
  const [audioMetadata, setAudioMetadata] = useState({})

  const handleFileUpload = (event) => {
    const file = event.target.files[0]
    if (file && (file.type === "audio/mpeg" || file.type === "audio/wav")) {
      setInputFile(file)
      setError(null)
      setIsProcessed(false)
      setSeparatedAudio({
        vocals: null,
        drums: null,
        bass: null,
        other: null,
        notes: { bass: null, drums: null, vocals: null },
        proc_id: null,
        song_name: null
      })
      setAudioMetadata({})
    } else {
      setError("Please upload a valid .mp3 or .wav file.")
      setInputFile(null)
    }
  }

  const separateAudio = async () => {
    if (!inputFile) return
    setIsProcessing(true)
    setError(null)

    const formData = new FormData()
    formData.append("file", inputFile)

    try {
      const response = await fetch("http://localhost:5000/process-audio", {
        method: "POST",
        body: formData,
      })

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`)
      }

      const data = await response.json()
      if (data.error) {
        throw new Error(data.error)
      }

      setSeparatedAudio({
        vocals: data.files.vocals,
        drums: data.files.drums,
        bass: data.files.bass,
        other: data.files.other,
        notes: data.notes,
        proc_id: data.proc_id,
        song_name: data.song_name
      })

      setAudioMetadata({
        key: data.metadata.key || "Unknown",
        tempo: data.metadata.tempo ? `${data.metadata.tempo} BPM` : "0.0 BPM"
      })

      setIsProcessing(false)
      setIsProcessed(true)
      setCurrentPage("results")
    } catch (err) {
      setError(`Failed to process audio: ${err.message}`)
      setIsProcessing(false)
    }
  }

  const AudioPlayer = ({ title, src, trackColor }) => {
    const audioRef = useRef(null)
    const [isPlaying, setIsPlaying] = useState(false)
    const [currentTime, setCurrentTime] = useState(0)
    const [duration, setDuration] = useState(0)
    const [volume, setVolume] = useState(1)
    const [isMuted, setIsMuted] = useState(false)

    useEffect(() => {
      const audio = audioRef.current
      audio.addEventListener("timeupdate", handleTimeUpdate)
      audio.addEventListener("loadedmetadata", handleLoadedMetadata)
      return () => {
        audio.removeEventListener("timeupdate", handleTimeUpdate)
        audio.removeEventListener("loadedmetadata", handleLoadedMetadata)
      }
    }, [])

    const handleTimeUpdate = () => {
      setCurrentTime(audioRef.current.currentTime)
    }

    const handleLoadedMetadata = () => {
      setDuration(audioRef.current.duration)
    }

    const togglePlay = () => {
      if (isPlaying) {
        audioRef.current.pause()
      } else {
        audioRef.current.play()
      }
      setIsPlaying(!isPlaying)
    }

    const handleSeek = (e) => {
      const seekTime = e.target.value
      audioRef.current.currentTime = seekTime
      setCurrentTime(seekTime)
    }

    const handleVolumeChange = (e) => {
      const newVolume = e.target.value
      setVolume(newVolume)
      audioRef.current.volume = newVolume
      setIsMuted(newVolume === 0)
    }

    const toggleMute = () => {
      if (isMuted) {
        audioRef.current.volume = volume
        setIsMuted(false)
      } else {
        audioRef.current.volume = 0
        setIsMuted(true)
      }
    }

    const formatTime = (time) => {
      const minutes = Math.floor(time / 60)
      const seconds = Math.floor(time % 60)
      return `${minutes}:${seconds.toString().padStart(2, "0")}`
    }

    const handleDownload = async () => {
      try {
        const response = await fetch(src)
        if (!response.ok) {
          throw new Error(`Failed to fetch WAV file: ${response.statusText}`)
        }
        const blob = await response.blob()
        const url = window.URL.createObjectURL(blob)
        const link = document.createElement("a")
        link.href = url
        link.download = `${title}.wav`
        document.body.appendChild(link)
        link.click()
        document.body.removeChild(link)
        window.URL.revokeObjectURL(url)
      } catch (err) {
        console.error(`Failed to download ${title}:`, err)
      }
    }

    return (
      <div style={styles.playerSection}>
        <div style={styles.playerControls}>
          <button onClick={togglePlay} style={styles.playButton} aria-label={isPlaying ? "Pause" : "Play"}>
            {isPlaying ? <Pause size={24} /> : <Play size={24} />}
          </button>
          <div style={styles.seekBarContainer}>
            <input
              type="range"
              min="0"
              max={duration}
              value={currentTime}
              onChange={handleSeek}
              style={{
                ...styles.seekBar,
                background: `linear-gradient(to right, ${trackColor} 0%, ${trackColor} ${(currentTime / duration) * 100}%, #2D3748 ${(currentTime / duration) * 100}%, #2D3748 100%)`,
              }}
              aria-label="Seek"
            />
            <span style={styles.timeDisplay}>
              {formatTime(currentTime)} / {formatTime(duration)}
            </span>
          </div>
        </div>
        <div style={styles.volumeAndDownload}>
          <div style={styles.volumeControl}>
            <button onClick={toggleMute} style={styles.muteButton} aria-label={isMuted ? "Unmute" : "Mute"}>
              {isMuted ? <VolumeX size={20} /> : <Volume2 size={20} />}
            </button>
            <input
              type="range"
              min="0"
              max="1"
              step="0.1"
              value={isMuted ? 0 : volume}
              onChange={handleVolumeChange}
              style={{
                ...styles.volumeBar,
                background: `linear-gradient(to right, ${trackColor} 0%, ${trackColor} ${volume * 100}%, #2D3748 ${volume * 100}%, #2D3748 100%)`,
              }}
              aria-label="Volume"
            />
          </div>
          <button onClick={handleDownload} style={styles.downloadButton} aria-label="Download">
            <Download size={20} />
            <span style={styles.downloadText}>Download</span>
          </button>
        </div>
        <audio ref={audioRef} src={src} />
      </div>
    )
  }

  const TrackContainer = ({ title, src, color, icon, notesUrl }) => {
    const handleViewPdf = () => {
      if (notesUrl) {
        window.open(notesUrl, '_blank')
      } else {
        console.error(`No notes URL for ${title}`)
      }
    }

    const handleDownloadPdf = async () => {
      if (!notesUrl) {
        console.error(`No notes URL for ${title}`)
        return
      }

      try {
        // Fetch both PDF and XML URLs
        const response = await fetch(`${notesUrl}?type=both`)
        if (!response.ok) {
          throw new Error(`Failed to fetch file URLs: ${response.statusText}`)
        }
        const data = await response.json()

        // Download PDF
        const pdfResponse = await fetch(data.pdf_url)
        if (!pdfResponse.ok) {
          throw new Error(`Failed to fetch PDF: ${pdfResponse.statusText}`)
        }
        const pdfBlob = await pdfResponse.blob()
        const pdfUrl = window.URL.createObjectURL(pdfBlob)
        const pdfLink = document.createElement("a")
        pdfLink.href = pdfUrl
        pdfLink.download = `notes-${title.toLowerCase()}.pdf`
        document.body.appendChild(pdfLink)
        pdfLink.click()
        document.body.removeChild(pdfLink)
        window.URL.revokeObjectURL(pdfUrl)

        // Download XML
        const xmlResponse = await fetch(data.xml_url)
        if (!xmlResponse.ok) {
          throw new Error(`Failed to fetch XML: ${xmlResponse.statusText}`)
        }
        const xmlBlob = await xmlResponse.blob()
        const xmlUrl = window.URL.createObjectURL(xmlBlob)
        const xmlLink = document.createElement("a")
        xmlLink.href = xmlUrl
        xmlLink.download = `notes-${title.toLowerCase()}.xml`
        document.body.appendChild(xmlLink)
        xmlLink.click()
        document.body.removeChild(xmlLink)
        window.URL.revokeObjectURL(xmlUrl)

      } catch (err) {
        console.error(`Failed to download files for ${title}:`, err)
      }
    }

    return (
      <div style={styles.trackContainer}>
        <div style={{ ...styles.trackHeader, backgroundColor: color }}>
          <div style={styles.trackTitleSection}>
            {icon}
            <h3 style={styles.trackTitle}>{title}</h3>
          </div>
          {title !== "Other" && (
            <div style={styles.pdfActions}>
              <button onClick={handleViewPdf} style={styles.viewPdfButton}>
                <FileText size={16} style={{ marginRight: "8px" }} />
                View Notes
              </button>
              <button onClick={handleDownloadPdf} style={styles.downloadPdfButton} aria-label={`Download ${title} notes`}>
                <Download size={16} />
              </button>
            </div>
          )}
        </div>
        <div style={styles.trackBody}>
          <AudioPlayer title={title} src={src} trackColor={color} />
        </div>
      </div>
    )
  }

  const PdfViewer = ({ track, url }) => {
    return (
      <div style={styles.pdfViewerContainer}>
        <div style={styles.pdfViewerHeader}>
          <button onClick={() => setViewingPdf(null)} style={styles.backButton}>
            <ArrowLeft size={20} style={{ marginRight: "8px" }} />
            Back to Audio
          </button>
          <h2 style={styles.pdfViewerTitle}>Notes for {track}</h2>
        </div>
        <iframe src={url} style={styles.pdfFrame} title={`Notes for ${track}`} />
      </div>
    )
  }

  const AnimatedWaveform = ({ color, gradient }) => {
    const svgRef = useRef(null)
    const [offset, setOffset] = useState(0)

    useEffect(() => {
      const animateWave = () => {
        setOffset((prevOffset) => (prevOffset + 1) % 500)
        requestAnimationFrame(animateWave)
      }
      const animationId = requestAnimationFrame(animateWave)
      return () => cancelAnimationFrame(animationId)
    }, [])

    return (
      <svg
        ref={svgRef}
        style={{ position: "absolute", inset: 0, width: "100%", height: "100%" }}
        preserveAspectRatio="none"
      >
        <defs>
          <linearGradient id={gradient} x1="0%" y1="50%" x2="100%" y2="50%">
            <stop offset="0%" stopColor={color} stopOpacity="0.2" />
            <stop offset="100%" stopColor={color} stopOpacity="0.8" />
          </linearGradient>
        </defs>
        <path
          d={`M-500 30 Q -485 10, -470 30 T -440 30 T -410 30 T -380 30 T -350 30 T -320 30 T -290 30 T -260 30 T -230 30 T -200 30 T -170 30 T -140 30 T -110 30 T -80 30 T -50 30 T -20 30 T 10 30 Q 25 50, 40 30 T 70 30 T 100 30 T 130 30 T 160 30 T 190 30 T 220 30 T 250 30 T 280 30 T 310 30 T 340 30 T 370 30 T 400 30 T 430 30 T 460 30 T 490 30`}
          stroke={color}
          strokeWidth="2"
          fill={`url(#${gradient})`}
          vectorEffect="non-scaling-stroke"
          transform={`translate(${offset}, 0)`}
        />
      </svg>
    )
  }

  const HomePage = () => {
    return (
      <>
        <h1 style={styles.heading}>Vocal Remover and Isolation</h1>
        <p style={styles.subheading}>Separate voice from music out of a song free with powerful AI algorithms</p>

        <div style={styles.waveformContainer}>
          <div style={styles.waveformRow}>
            <span style={{ ...styles.waveformLabel, color: "#10B981" }}>Music</span>
            <svg style={styles.waveformIcon} viewBox="0 0 24 24" fill="none">
              <path d="M12 3v17M19 6v11M5 6v11" stroke="#10B981" strokeWidth="2" />
            </svg>
            <div style={styles.waveform}>
              <AnimatedWaveform color="#10B981" gradient="greenGradient" />
            </div>
          </div>

          <div style={styles.waveformRow}>
            <span style={{ ...styles.waveformLabel, color: "#A855F7" }}>Vocal</span>
            <svg style={styles.waveformIcon} viewBox="0 0 24 24" fill="none">
              <path d="M12 3v17M19 6v11M5 6v11" stroke="#A855F7" strokeWidth="2" />
            </svg>
            <div style={styles.waveform}>
              <AnimatedWaveform color="#A855F7" gradient="purpleGradient" />
            </div>
          </div>
        </div>

        {!inputFile && !isProcessing && (
          <>
            <input
              type="file"
              accept=".mp3,.wav"
              onChange={handleFileUpload}
              id="file-upload"
              style={styles.fileInput}
            />
            <label htmlFor="file-upload" style={styles.uploadButton}>
              Browse my files
            </label>
            {error && <p style={styles.error}>{error}</p>}
          </>
        )}

        {isProcessing && (
          <div>
            <div style={styles.loader}></div>
            <p>Processing your audio... This might take a while!</p>
          </div>
        )}

        {inputFile && !isProcessing && !isProcessed && (
          <button onClick={separateAudio} style={styles.uploadButton}>
            Process Audio
          </button>
        )}

        {inputFile && !isProcessing && !isProcessed && (
          <div style={styles.previewContainer}>
            <h2 style={styles.previewTitle}>Audio Preview</h2>
            <div style={styles.originalAudioPlayer}>
              <h3 style={styles.audioTitle}>Original Input</h3>
              <AudioPlayer title="Original Input" src={URL.createObjectURL(inputFile)} trackColor="#A855F7" />
            </div>
          </div>
        )}
      </>
    )
  }

  const ResultsPage = () => {
    if (!isProcessed) {
      return (
        <div style={styles.noResults}>
          <h2>No Results Available</h2>
          <p>Please process an audio file first to see results.</p>
          <button onClick={() => setCurrentPage("home")} style={styles.uploadButton}>
            Go to Home
          </button>
        </div>
      )
    }

    const trackColors = {
      vocals: "#A855F7",
      drums: "#10B981",
      bass: "#A855F7",
      other: "#10B981",
    }

    const trackIcons = {
      vocals: <Music size={24} color={trackColors.vocals} />,
      drums: <Music size={24} color={trackColors.drums} />,
      bass: <Music size={24} color={trackColors.bass} />,
      other: <Music size={24} color={trackColors.other} />,
    }

    return (
      <>
        <h1 style={styles.heading}>Processing Results</h1>
        <p style={styles.subheading}>Your audio has been successfully processed</p>

        <div style={styles.resultsContainer}>
          <div style={styles.inputSection}>
            <h2 style={styles.sectionTitle}>Original Audio</h2>
            <div style={styles.originalAudioPlayer}>
              <h3 style={styles.audioTitle}>Original Input</h3>
              <AudioPlayer title="Original Input" src={URL.createObjectURL(inputFile)} trackColor="#A855F7" />
            </div>

            <div style={styles.metadataContainer}>
              <div style={styles.metadataItem}>
                <span style={styles.metadataLabel}>Key</span>
                <span style={styles.metadataValue}>{audioMetadata.key}</span>
              </div>
              <div style={styles.metadataItem}>
                <span style={styles.metadataLabel}>Tempo</span>
                <span style={styles.metadataValue}>{audioMetadata.tempo}</span>
              </div>
            </div>
          </div>

          <h2 style={styles.sectionTitle}>Separated Tracks</h2>

          <TrackContainer
            title="Vocals"
            src={separatedAudio.vocals}
            color={trackColors.vocals}
            icon={trackIcons.vocals}
            notesUrl={separatedAudio.notes.vocals}
          />

          <div style={styles.trackSeparator}></div>

          <TrackContainer
            title="Drums"
            src={separatedAudio.drums}
            color={trackColors.drums}
            icon={trackIcons.drums}
            notesUrl={separatedAudio.notes.drums}
          />

          <div style={styles.trackSeparator}></div>

          <TrackContainer
            title="Bass"
            src={separatedAudio.bass}
            color={trackColors.bass}
            icon={trackIcons.bass}
            notesUrl={separatedAudio.notes.bass}
          />

          <div style={styles.trackSeparator}></div>

          <TrackContainer
            title="Other"
            src={separatedAudio.other}
            color={trackColors.other}
            icon={trackIcons.other}
          />
        </div>
      </>
    )
  }

  const styles = {
    container: {
      minHeight: "100vh",
      backgroundColor: "#0F0F13",
      color: "white",
      fontFamily: "Arial, sans-serif",
      display: "flex",
      flexDirection: "column",
      alignItems: "center",
    },
    main: {
      maxWidth: "900px",
      margin: "0 auto",
      padding: "24px 16px",
      textAlign: "center",
      width: "100%",
    },
    heading: {
      fontSize: "48px",
      fontWeight: "bold",
      marginBottom: "24px",
      background: "linear-gradient(to right, #A855F7, #10B981)",
      WebkitBackgroundClip: "text",
      WebkitTextFillColor: "transparent",
    },
    subheading: {
      fontSize: "20px",
      color: "#E9D8FD",
      marginBottom: "64px",
    },
    waveformContainer: {
      maxWidth: "600px",
      margin: "0 auto 64px",
    },
    waveformRow: {
      display: "flex",
      alignItems: "center",
      gap: "16px",
      marginBottom: "16px",
    },
    waveformLabel: {
      width: "64px",
      textAlign: "left",
    },
    waveformIcon: {
      width: "24px",
      opacity: "0.6",
    },
    waveform: {
      flexGrow: 1,
      height: "48px",
      backgroundColor: "#1A1A23",
      borderRadius: "4px",
      overflow: "hidden",
      position: "relative",
    },
    uploadButton: {
      padding: "12px 24px",
      borderRadius: "9999px",
      border: "1px solid #A855F7",
      backgroundColor: "#1A1A23",
      color: "#E9D8FD",
      fontSize: "16px",
      cursor: "pointer",
      transition: "all 0.3s ease",
    },
    fileInput: {
      display: "none",
    },
    error: {
      color: "#EF4444",
      marginTop: "12px",
    },
    loader: {
      border: "4px solid #A855F7",
      borderTop: "4px solid #10B981",
      borderRadius: "50%",
      width: "40px",
      height: "40px",
      animation: "spin 1s linear infinite",
      margin: "20px auto",
    },
    trackContainer: {
      width: "100%",
      backgroundColor: "#1A1A23",
      borderRadius: "12px",
      overflow: "hidden",
      marginBottom: "16px",
      boxShadow: "0 4px 6px rgba(0, 0, 0, 0.1)",
    },
    trackHeader: {
      display: "flex",
      justifyContent: "space-between",
      alignItems: "center",
      padding: "16px 24px",
      borderTopLeftRadius: "12px",
      borderTopRightRadius: "12px",
    },
    trackTitleSection: {
      display: "flex",
      alignItems: "center",
      gap: "12px",
    },
    trackTitle: {
      margin: 0,
      fontSize: "20px",
      fontWeight: "bold",
      color: "white",
    },
    trackBody: {
      padding: "20px 24px",
    },
    playerSection: {
      display: "flex",
      flexDirection: "column",
      gap: "16px",
    },
    playerControls: {
      display: "flex",
      alignItems: "center",
      gap: "16px",
      width: "100%",
    },
    playButton: {
      backgroundColor: "#2D2D38",
      border: "none",
      color: "white",
      cursor: "pointer",
      width: "48px",
      height: "48px",
      borderRadius: "50%",
      display: "flex",
      alignItems: "center",
      justifyContent: "center",
      flexShrink: 0,
    },
    seekBarContainer: {
      flexGrow: 1,
      display: "flex",
      flexDirection: "column",
      gap: "8px",
    },
    seekBar: {
      width: "100%",
      height: "6px",
      WebkitAppearance: "none",
      background: "#2D3748",
      outline: "none",
      borderRadius: "3px",
      cursor: "pointer",
    },
    timeDisplay: {
      fontSize: "14px",
      color: "#A0AEC0",
      alignSelf: "flex-end",
    },
    volumeAndDownload: {
      display: "flex",
      justifyContent: "space-between",
      alignItems: "center",
    },
    volumeControl: {
      display: "flex",
      alignItems: "center",
      gap: "8px",
    },
    muteButton: {
      backgroundColor: "transparent",
      border: "none",
      color: "white",
      cursor: "pointer",
    },
    volumeBar: {
      width: "100px",
      height: "4px",
      WebkitAppearance: "none",
      background: "#2D3748",
      outline: "none",
      borderRadius: "2px",
      cursor: "pointer",
    },
    downloadButton: {
      display: "flex",
      alignItems: "center",
      gap: "8px",
      backgroundColor: "#2D2D38",
      border: "none",
      color: "white",
      padding: "8px 16px",
      borderRadius: "4px",
      cursor: "pointer",
      transition: "background-color 0.2s",
    },
    downloadText: {
      fontSize: "14px",
    },
    pdfActions: {
      display: "flex",
      gap: "8px",
    },
    viewPdfButton: {
      display: "flex",
      alignItems: "center",
      backgroundColor: "rgba(255, 255, 255, 0.2)",
      border: "none",
      color: "white",
      borderRadius: "4px",
      padding: "8px 12px",
      fontSize: "14px",
      cursor: "pointer",
      transition: "background-color 0.2s",
    },
    downloadPdfButton: {
      display: "flex",
      alignItems: "center",
      justifyContent: "center",
      backgroundColor: "rgba(255, 255, 255, 0.2)",
      border: "none",
      color: "white",
      borderRadius: "4px",
      padding: "8px",
      cursor: "pointer",
      transition: "background-color 0.2s",
    },
    pdfViewerContainer: {
      position: "fixed",
      top: 0,
      left: 0,
      right: 0,
      bottom: 0,
      backgroundColor: "#0F0F13",
      zIndex: 1000,
      padding: "24px",
      display: "flex",
      flexDirection: "column",
    },
    pdfViewerHeader: {
      display: "flex",
      alignItems: "center",
      marginBottom: "16px",
    },
    backButton: {
      display: "flex",
      alignItems: "center",
      backgroundColor: "transparent",
      border: "1px solid #A855F7",
      color: "#A855F7",
      borderRadius: "4px",
      padding: "8px 16px",
      fontSize: "14px",
      cursor: "pointer",
      marginRight: "16px",
    },
    pdfViewerTitle: {
      color: "#E9D8FD",
      fontSize: "24px",
      margin: 0,
    },
    pdfFrame: {
      flex: 1,
      border: "none",
      borderRadius: "8px",
      backgroundColor: "#fff",
    },
    previewContainer: {
      marginTop: "32px",
      backgroundColor: "#1A1A23",
      borderRadius: "12px",
      padding: "24px",
      width: "100%",
    },
    previewTitle: {
      color: "#E9D8FD",
      fontSize: "24px",
      marginBottom: "16px",
      textAlign: "left",
    },
    resultsContainer: {
      display: "flex",
      flexDirection: "column",
      gap: "32px",
      width: "100%",
    },
    inputSection: {
      backgroundColor: "#1A1A23",
      borderRadius: "12px",
      padding: "24px",
      width: "100%",
      marginBottom: "32px",
    },
    sectionTitle: {
      color: "#E9D8FD",
      fontSize: "24px",
      marginBottom: "24px",
      textAlign: "left",
    },
    metadataContainer: {
      display: "flex",
      justifyContent: "center",
      gap: "32px",
      marginTop: "24px",
      padding: "16px",
      backgroundColor: " #2D2D38",
      borderRadius: "8px",
    },
    metadataItem: {
      display: "flex",
      flexDirection: "column",
      alignItems: "center",
    },
    metadataLabel: {
      color: "#A0AEC0",
      fontSize: "14px",
      marginBottom: "4px",
    },
    metadataValue: {
      color: "#10B981",
      fontSize: "18px",
      fontWeight: "bold",
    },
    noResults: {
      display: "flex",
      flexDirection: "column",
      alignItems: "center",
      justifyContent: "center",
      gap: "16px",
      marginTop: "64px",
      padding: "32px",
      backgroundColor: "#1A1A23",
      borderRadius: "12px",
      width: "100%",
    },
    originalAudioPlayer: {
      backgroundColor: "#2D2D38",
      borderRadius: "8px",
      padding: "16px",
    },
    audioTitle: {
      color: "#E9D8FD",
      marginBottom: "16px",
      fontSize: "18px",
      textAlign: "center",
    },
    trackSeparator: {
      height: "1px",
      backgroundColor: "#2D3748",
      margin: "8px 0 24px 0",
      width: "100%",
    },
  }

  if (viewingPdf) {
    return (
      <div style={styles.container}>
        <PdfViewer track={viewingPdf.track} url={viewingPdf.url} />
      </div>
    )
  }

  return (
    <div style={styles.container}>
      <main style={styles.main}>
        {currentPage === "home" && <HomePage />}
        {currentPage === "results" && <ResultsPage />}
      </main>
    </div>
  )
}

export default App