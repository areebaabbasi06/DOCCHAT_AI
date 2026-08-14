const API_URL = "https://docchat-ai-production.up.railway.app";
let currentFile = null;
let currentImage = null;
let sessionId = crypto.randomUUID();

let isProcessing = false;
let isSending = false;


// ==========================================
// DOM ELEMENTS
// ==========================================

const pdfInput =
    document.getElementById("pdfInput");

const fileName =
    document.getElementById("fileName");

const fileSize =
    document.getElementById("fileSize");

const documentInfo =
    document.getElementById("documentInfo");

const progressBar =
    document.getElementById("progressBar");

const processingStatus =
    document.getElementById("processingStatus");

const chatMessages =
    document.getElementById("chatMessages");

const questionInput =
    document.getElementById("questionInput");


// ==========================================
// IMAGE ELEMENTS
// ==========================================

const imageInput =
    document.getElementById("imageInput");

const imagePreview =
    document.getElementById("imagePreview");

const imagePreviewImg =
    document.getElementById("imagePreviewImg");

const imageFileName =
    document.getElementById("imageFileName");

const imageButton =
    document.getElementById("imageButton");


// ==========================================
// PDF VIEWER
// ==========================================

const pdfViewerContainer =
    document.getElementById("pdfViewerContainer");

const pdfViewer =
    document.getElementById("pdfViewer");


// ==========================================
// PDF UPLOAD
// ==========================================

if (pdfInput) {

    pdfInput.addEventListener(
        "change",
        function () {

            const file =
                this.files[0];

            if (!file) {
                return;
            }

            // 5 MB LIMIT
            const maxSize =
                5 * 1024 * 1024;

            if (file.size > maxSize) {

                alert(
                    "File size must be 5 MB or less."
                );

                pdfInput.value = "";

                return;
            }


            // PDF CHECK
            if (
                file.type !== "application/pdf" &&
                !file.name
                    .toLowerCase()
                    .endsWith(".pdf")
            ) {

                alert(
                    "Please select a PDF file."
                );

                pdfInput.value = "";

                return;
            }


            // SAVE FILE
            currentFile = file;


            if (fileName) {

                fileName.textContent =
                    file.name;
            }


            if (fileSize) {

                fileSize.textContent =
                    formatFileSize(
                        file.size
                    );
            }


            if (documentInfo) {

                documentInfo.style.display =
                    "flex";
            }


            // PROCESS PDF
            processDocument(file);
        }
    );
}


// ==========================================
// IMAGE ATTACHMENT
// ==========================================

if (imageInput) {

    imageInput.addEventListener(
        "change",
        function () {

            const file =
                this.files[0];

            if (!file) {
                return;
            }


            // 5 MB IMAGE LIMIT
            const maxImageSize =
                5 * 1024 * 1024;

            if (
                file.size >
                maxImageSize
            ) {

                alert(
                    "Image size must be 5 MB or less."
                );

                imageInput.value = "";

                return;
            }


            // IMAGE CHECK
            if (
                !file.type.startsWith("image/")
            ) {

                alert(
                    "Please select an image file."
                );

                imageInput.value = "";

                return;
            }


            // SAVE IMAGE
            currentImage = file;


            // FILE NAME
            if (imageFileName) {

                imageFileName.textContent =
                    file.name;
            }


            // IMAGE PREVIEW
            const reader =
                new FileReader();

            reader.onload =
                function (event) {

                    if (imagePreviewImg) {

                        imagePreviewImg.src =
                            event.target.result;
                    }


                    if (imagePreview) {

                        imagePreview.style.display =
                            "flex";
                    }
                };


            reader.readAsDataURL(file);


            // ACTIVE PAPERCLIP
            if (imageButton) {

                imageButton.classList.add(
                    "active"
                );
            }
        }
    );
}


// ==========================================
// REMOVE IMAGE
// ==========================================

function removeImage() {

    currentImage = null;


    if (imageInput) {

        imageInput.value = "";
    }


    if (imagePreviewImg) {

        imagePreviewImg.src = "";
    }


    if (imageFileName) {

        imageFileName.textContent = "";
    }


    if (imagePreview) {

        imagePreview.style.display =
            "none";
    }


    if (imageButton) {

        imageButton.classList.remove(
            "active"
        );
    }
}


// ==========================================
// PROCESS PDF
// ==========================================

async function processDocument(file) {

    if (isProcessing) {

        return;
    }


    isProcessing = true;


    if (processingStatus) {

        processingStatus.textContent =
            "Uploading document...";
    }


    if (progressBar) {

        progressBar.style.width =
            "20%";
    }


    const formData =
        new FormData();


    formData.append(
        "file",
        file
    );


    formData.append(
        "session_id",
        sessionId
    );


    try {

        // UPLOAD PDF
        const response =
            await fetch(
                `${API_URL}/upload`,
                {
                    method: "POST",
                    body: formData
                }
            );


        if (!response.ok) {

            throw new Error(
                `Upload failed (${response.status})`
            );
        }


        if (progressBar) {

            progressBar.style.width =
                "70%";
        }


        const data =
            await response.json();


        // BACKEND ERROR
        if (
            data.status !==
            "success"
        ) {

            throw new Error(
                data.message ||
                "PDF processing failed."
            );
        }


        // PREVIEW
        if (progressBar) {

            progressBar.style.width =
                "90%";
        }


        if (processingStatus) {

            processingStatus.textContent =
                "Preparing document preview...";
        }


        const pdfUrl =
            `${API_URL}/pdf/${encodeURIComponent(
                file.name
            )}`;


        if (pdfViewer) {

            pdfViewer.src =
                pdfUrl;
        }


        if (pdfViewerContainer) {

            pdfViewerContainer.style.display =
                "block";
        }


        // PAGE COUNT
        const pageCount =
            document.getElementById(
                "pageCount"
            );


        if (
            pageCount &&
            data.pages !== undefined
        ) {

            pageCount.textContent =
                data.pages;
        }


        // CHUNK COUNT
        const chunkCount =
            document.getElementById(
                "chunkCount"
            );


        if (
            chunkCount &&
            data.chunks !== undefined
        ) {

            chunkCount.textContent =
                data.chunks;
        }


        // COMPLETE
        if (progressBar) {

            progressBar.style.width =
                "100%";
        }


        if (processingStatus) {

            processingStatus.textContent =
                "Document processed successfully ✓";
        }


        console.log(
            "PDF ready:",
            data.pages,
            "pages,",
            data.chunks,
            "chunks"
        );


    } catch (error) {

        console.error(
            "PDF processing error:",
            error
        );


        currentFile = null;


        if (progressBar) {

            progressBar.style.width =
                "0%";
        }


        if (processingStatus) {

            processingStatus.textContent =
                "Could not process document.";
        }


        if (pdfViewer) {

            pdfViewer.src = "";
        }


        if (pdfViewerContainer) {

            pdfViewerContainer.style.display =
                "none";
        }


        alert(
            "PDF process nahi ho saki.\n\n" +
            error.message
        );


    } finally {

        isProcessing = false;
    }
}


// ==========================================
// SEND MESSAGE
// ==========================================

async function sendMessage() {

    if (isSending) {

        return;
    }


    const question =
        questionInput
            ? questionInput.value.trim()
            : "";


    // Text bhi nahi aur image bhi nahi
    if (
        !question &&
        !currentImage
    ) {

        return;
    }


    // PDF OPTIONAL
    const usePdf =
        currentFile !== null;


    // USER TEXT MESSAGE
    if (question) {

        addMessage(
            question,
            "user"
        );
    }


    // USER IMAGE MESSAGE
    if (currentImage) {

        addImageMessage(
            currentImage
        );
    }


    // CLEAR TEXT INPUT
    if (questionInput) {

        questionInput.value = "";
    }


    // LOADING MESSAGE
    const loadingMessage =
        addMessage(
            "Thinking...",
            "ai"
        );


    isSending = true;


    if (questionInput) {

        questionInput.disabled =
            true;
    }


    if (imageButton) {

        imageButton.disabled =
            true;
    }


    try {

        // ==================================
        // CONVERT IMAGE TO DATA URL
        // ==================================

        let imageData = null;


        if (currentImage) {

            imageData =
                await fileToDataURL(
                    currentImage
                );
        }


        // ==================================
        // SEND TO BACKEND
        // ==================================

        const response =
            await fetch(
                `${API_URL}/chat`,
                {
                    method: "POST",

                    headers: {
                        "Content-Type":
                            "application/json"
                    },

                    body:
                        JSON.stringify({

                            question:
                                question,

                            session_id:
                                sessionId,

                            use_pdf:
                                usePdf,

                            image:
                                imageData

                        })
                }
            );


        if (!response.ok) {

            throw new Error(
                `Chat request failed (${response.status})`
            );
        }


        const data =
            await response.json();


        // REMOVE THINKING
        if (loadingMessage) {

            loadingMessage.remove();
        }


        // BACKEND ERROR
        if (
            data.status ===
            "error"
        ) {

            addMessage(
                data.message ||
                "An error occurred.",
                "ai"
            );

            return;
        }


        // ==================================
        // ANSWER
        // ==================================

        const answer =
            data.answer ||
            data.response ||
            "No answer received.";


        addMessage(
            answer,
            "ai"
        );


        // ==================================
        // SOURCES
        // ==================================

        if (
            data.sources &&
            Array.isArray(
                data.sources
            )
        ) {

            showSources(
                data.sources
            );
        }


        // Clear image after success
        removeImage();


    } catch (error) {

        console.error(
            "Chat error:",
            error
        );


        if (loadingMessage) {

            loadingMessage.remove();
        }


        addMessage(
            "Sorry, I could not connect to the backend.\n\n" +
            error.message,
            "ai"
        );


    } finally {

        isSending = false;


        if (questionInput) {

            questionInput.disabled =
                false;

            questionInput.focus();
        }


        if (imageButton) {

            imageButton.disabled =
                false;
        }
    }
}


// ==========================================
// IMAGE MESSAGE IN CHAT
// ==========================================

function addImageMessage(file) {

    const message =
        document.createElement(
            "div"
        );


    message.classList.add(
        "message",
        "user-message"
    );


    const imageUrl =
        URL.createObjectURL(
            file
        );


    message.innerHTML = `

        <div class="message-avatar user-avatar">

            <i class="fa-solid fa-user"></i>

        </div>


        <div class="message-body">

            <div class="message-header">

                <span class="message-name">
                    You
                </span>

                <span class="message-time">
                    Now
                </span>

            </div>


            <div class="message-text">

                <img
                    src="${imageUrl}"
                    alt="Attached image"
                    class="chat-image"
                >

            </div>

        </div>
    `;


    chatMessages.appendChild(
        message
    );


    chatMessages.scrollTop =
        chatMessages.scrollHeight;
}


// ==========================================
// FILE TO DATA URL
// ==========================================

function fileToDataURL(file) {

    return new Promise(
        function (
            resolve,
            reject
        ) {

            const reader =
                new FileReader();


            reader.onload =
                function () {

                    resolve(
                        reader.result
                    );
                };


            reader.onerror =
                function () {

                    reject(
                        new Error(
                            "Could not read image."
                        )
                    );
                };


            reader.readAsDataURL(
                file
            );
        }
    );
}


// ==========================================
// ADD MESSAGE
// ==========================================

function addMessage(
    text,
    sender
) {

    const message =
        document.createElement(
            "div"
        );


    message.classList.add(
        "message",
        sender === "ai"
            ? "ai-message"
            : "user-message"
    );


    const icon =
        sender === "ai"
            ? "fa-robot"
            : "fa-user";


    const name =
        sender === "ai"
            ? "DocChat AI"
            : "You";


    // ======================================
    // FORMAT MESSAGE
    // ======================================

    let formattedText;


    if (sender === "ai") {

        formattedText =
            formatAnswer(
                String(text)
            );

    } else {

        formattedText =
            escapeHTML(
                String(text)
            ).replace(
                /\n/g,
                "<br>"
            );
    }


    // ======================================
    // MESSAGE HTML
    // ======================================

    message.innerHTML = `

        <div class="message-avatar ${
            sender === "ai"
                ? "ai-avatar"
                : "user-avatar"
        }">

            <i class="fa-solid ${icon}"></i>

        </div>


        <div class="message-body">

            <div class="message-header">

                <span class="message-name">
                    ${name}
                </span>

                <span class="message-time">
                    Now
                </span>

            </div>


            <div class="message-text">

                ${formattedText}

            </div>

        </div>

    `;


    chatMessages.appendChild(
        message
    );


    chatMessages.scrollTop =
        chatMessages.scrollHeight;


    return message;
}


// ==========================================
// FORMAT AI ANSWER
// ==========================================

function formatAnswer(text) {

    if (!text) {

        return "";
    }


    // Normalize line breaks
    let answer =
        String(text)
            .replace(
                /\r\n/g,
                "\n"
            )
            .replace(
                /\r/g,
                "\n"
            );


    // Escape HTML first
    answer =
        escapeHTML(
            answer
        );


    // ======================================
    // MARKDOWN HEADINGS
    // ======================================

    answer =
        answer.replace(
            /^######\s+(.*?)$/gm,
            "<h6>$1</h6>"
        );


    answer =
        answer.replace(
            /^#####\s+(.*?)$/gm,
            "<h5>$1</h5>"
        );


    answer =
        answer.replace(
            /^####\s+(.*?)$/gm,
            "<h4>$1</h4>"
        );


    answer =
        answer.replace(
            /^###\s+(.*?)$/gm,
            "<h3>$1</h3>"
        );


    answer =
        answer.replace(
            /^##\s+(.*?)$/gm,
            "<h3>$1</h3>"
        );


    answer =
        answer.replace(
            /^#\s+(.*?)$/gm,
            "<h3>$1</h3>"
        );


    // ======================================
    // BOLD
    // ======================================

    answer =
        answer.replace(
            /\*\*(.*?)\*\*/g,
            "<strong>$1</strong>"
        );


    // ======================================
    // ITALIC
    // ======================================

    answer =
        answer.replace(
            /(?<!\*)\*([^*\n]+)\*(?!\*)/g,
            "<em>$1</em>"
        );


    // ======================================
    // INLINE CODE
    // ======================================

    answer =
        answer.replace(
            /`([^`\n]+)`/g,
            "<code>$1</code>"
        );


    // ======================================
    // LISTS
    // ======================================

    const lines =
        answer.split("\n");


    let output = [];

    let listType = null;


    function closeList() {

        if (listType === "ul") {

            output.push("</ul>");

        }

        if (listType === "ol") {

            output.push("</ol>");
        }

        listType = null;
    }


    for (
        let i = 0;
        i < lines.length;
        i++
    ) {

        const line =
            lines[i];


        // BULLET
        const bulletMatch =
            line.match(
                /^\s*(?:[-*•])\s+(.*)$/
            );


        if (bulletMatch) {

            if (listType !== "ul") {

                closeList();

                output.push(
                    "<ul>"
                );

                listType =
                    "ul";
            }


            output.push(
                `<li>${bulletMatch[1]}</li>`
            );


            continue;
        }


        // NUMBERED LIST
        const numberMatch =
            line.match(
                /^\s*\d+\.\s+(.*)$/
            );


        if (numberMatch) {

            if (listType !== "ol") {

                closeList();

                output.push(
                    "<ol>"
                );

                listType =
                    "ol";
            }


            output.push(
                `<li>${numberMatch[1]}</li>`
            );


            continue;
        }


        // NORMAL LINE
        closeList();

        output.push(
            line
        );
    }


    closeList();


    answer =
        output.join("\n");


    // ======================================
    // PARAGRAPHS
    // ======================================

    const parts =
        answer.split(
            /\n\s*\n/
        );


    let result = [];


    parts.forEach(
        function (part) {

            part =
                part.trim();


            if (!part) {

                return;
            }


            // Don't wrap HTML blocks
            if (
                part.startsWith(
                    "<h3>"
                ) ||
                part.startsWith(
                    "<h4>"
                ) ||
                part.startsWith(
                    "<h5>"
                ) ||
                part.startsWith(
                    "<h6>"
                ) ||
                part.startsWith(
                    "<ul>"
                ) ||
                part.startsWith(
                    "<ol>"
                )
            ) {

                result.push(
                    part
                );

            } else {

                part =
                    part.replace(
                        /\n/g,
                        "<br>"
                    );


                result.push(
                    `<p>${part}</p>`
                );
            }
        }
    );


    return result.join("");
}


// ==========================================
// SHOW SOURCES
// ==========================================

function showSources(sources) {

    const sourcesContainer =
        document.getElementById(
            "sources"
        );


    if (!sourcesContainer) {

        return;
    }


    sourcesContainer.innerHTML =
        "";


    if (
        !Array.isArray(
            sources
        )
    ) {

        return;
    }


    sources.forEach(
        function (source) {

            const sourceItem =
                document.createElement(
                    "div"
                );


            sourceItem.className =
                "source-item";


            let sourceText =
                "";


            if (
                typeof source ===
                    "object" &&
                source !== null
            ) {

                sourceText =
                    source.title ||
                    source.url ||
                    source.content ||
                    "Web Source";

            } else {

                sourceText =
                    String(source);
            }


            sourceItem.innerHTML = `

                <i class="fa-solid fa-globe"></i>

                ${escapeHTML(
                    sourceText
                )}

            `;


            sourcesContainer.appendChild(
                sourceItem
            );
        }
    );
}


// ==========================================
// REMOVE PDF
// ==========================================

function removeFile() {

    currentFile = null;


    if (pdfInput) {

        pdfInput.value = "";
    }


    if (fileName) {

        fileName.textContent =
            "No document selected";
    }


    if (fileSize) {

        fileSize.textContent =
            "—";
    }


    if (documentInfo) {

        documentInfo.style.display =
            "none";
    }


    if (progressBar) {

        progressBar.style.width =
            "0%";
    }


    if (processingStatus) {

        processingStatus.textContent =
            "Waiting for document...";
    }


    const pageCount =
        document.getElementById(
            "pageCount"
        );


    if (pageCount) {

        pageCount.textContent =
            "0";
    }


    const chunkCount =
        document.getElementById(
            "chunkCount"
        );


    if (chunkCount) {

        chunkCount.textContent =
            "0";
    }


    if (pdfViewer) {

        pdfViewer.src =
            "";
    }


    if (pdfViewerContainer) {

        pdfViewerContainer.style.display =
            "none";
    }
}


// ==========================================
// NEW CHAT
// ==========================================

function newChat() {

    sessionId =
        crypto.randomUUID();


    // Remove selected image
    removeImage();


    if (chatMessages) {

        chatMessages.innerHTML =
            "";
    }


    const sourcesContainer =
        document.getElementById(
            "sources"
        );


    if (sourcesContainer) {

        sourcesContainer.innerHTML =
            "";
    }


    addMessage(
        "New chat started. You can ask questions with or without a PDF.",
        "ai"
    );
}


// ==========================================
// ENTER TO SEND
// ==========================================

if (questionInput) {

    questionInput.addEventListener(
        "keydown",
        function (event) {

            if (
                event.key === "Enter" &&
                !event.shiftKey
            ) {

                event.preventDefault();

                sendMessage();
            }
        }
    );
}


// ==========================================
// FILE SIZE
// ==========================================

function formatFileSize(
    bytes
) {

    if (
        bytes < 1024
    ) {

        return (
            bytes +
            " B"
        );
    }


    if (
        bytes <
        1024 * 1024
    ) {

        return (
            (
                bytes /
                1024
            ).toFixed(1) +
            " KB"
        );
    }


    return (
        (
            bytes /
            (
                1024 *
                1024
            )
        ).toFixed(2) +
        " MB"
    );
}


// ==========================================
// SECURITY
// ==========================================

function escapeHTML(
    text
) {

    const div =
        document.createElement(
            "div"
        );


    div.textContent =
        text;


    return div.innerHTML;
}