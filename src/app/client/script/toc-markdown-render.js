// toc-markdown-render.js


class TocMarkdownRender {

    constructor() {
        this.editor = null; // Will be set by EasyMDE
        this.preview = document.getElementById('preview-area');
        this.filter = document.getElementById('filterInput');
        this.treeInstance = null;

        this.init();
    }

    init() {
        // Initialize EasyMDE
        this.editor = new EasyMDE({
            element: document.getElementById('markdown-editor'),
            autoDownloadFontAwesome: false,
            spellChecker: false,
            status: false,
            toolbar: [
                "bold", "italic", "strikethrough", "|",
                "heading-1", "heading-2", "heading-3", "heading-smaller", "heading-bigger", "|",
                "code", "quote", "unordered-list", "ordered-list", "|",
                "link", "image", "|",
                "preview", "side-by-side", "fullscreen", "|",
                "guide"
            ],
        });

        // Initial Render
        this.refresh();

        // Real-time update on input
        this.editor.codemirror.on("change", () => this.refresh());

        // Filtering logic
        this.filter.addEventListener('input', (e) => {
            if (this.treeInstance) {
                this.treeInstance.filterNodes(e.target.value, { autoExpand: true });
            }
        });
    }

    refresh() {
        // 1. Get markdown content from EasyMDE
        const markdownContent = this.editor.value();
        
        // 2. Convert MD to HTML
        const htmlContent = marked.parse(markdownContent);
        this.preview.innerHTML = htmlContent;

        // 3. Map Headers to IDs
        const headers = Array.from(this.preview.querySelectorAll('h1, h2, h3, h4, h5, h6'));
        headers.forEach((h, i) => h.id = `nav-header-${i}`);

        // 4. Build Tree Structure
        const treeData = this.buildTree(headers);

        // 5. Update Tree - Using reload() for total reactivity
        if (!this.treeInstance) {
            this.initTree(treeData);
        } else {
            this.treeInstance.load(treeData);
        }
    }

    buildTree(headers) {
        const root = [];
        const stack = [{ level: 0, children: root }];

        headers.forEach(h => {
            const level = parseInt(h.tagName.substring(1));
            const node = {
                title: h.innerText,
                key: h.id,
                expanded: true, // Keep it expanded while typing
                children: []
            };

            while (stack.length > 1 && stack[stack.length - 1].level >= level) {
                stack.pop();
            }

            stack[stack.length - 1].children.push(node);
            stack.push({ level, children: node.children });
        });
        return root;
    }

    initTree(data) {
        this.treeInstance = new mar10.Wunderbaum({
            element: document.getElementById('toc-tree'),
            source: data,
            iconMap: {
                folder: "fa-solid fa-folder",
                folderOpen: "fa-solid fa-folder-open",
                doc: "fa-regular fa-file",
                expanderExpanded: "fa-solid fa-chevron-down",
                expanderCollapsed: "fa-solid fa-chevron-right",
            },
            header: false,
            connectLines: true,
            fixedCol: false,
            enhance: (e) => {
                // Force folder icon if node has children
                if (e.node.children && e.node.children.length > 0) {
                    e.node.folder = true;
                }
            },
            click: (e) => {
                const target = document.getElementById(e.node.key);
                if (target) {
                    target.scrollIntoView({ behavior: 'smooth', block: 'start' });
                }
            }
        });
    }
}
