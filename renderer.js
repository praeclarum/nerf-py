function startRenderer($imgElement, initialCameraTransform) {
    const renderer = new Renderer($imgElement, initialCameraTransform);
    renderer.start();
    return renderer;
}

async function delay(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
}

class Renderer {
    constructor($img, initialCameraTransform) {
        this.$img = $img;
        this.cameraTransform = initialCameraTransform;
        this.shouldRun = false;
        // Want to override mouse drag to be rotation so handle those events on $img
        $img.addEventListener('mousedown', (e) => {
            this.isDragging = true;
            this.dragStart = { x: e.clientX, y: e.clientY };
            e.preventDefault();
        });
        $img.addEventListener('mouseup', (e) => {
            this.isDragging = false;
            e.preventDefault();
        });
        $img.addEventListener('mousemove', (e) => {
            if (this.isDragging) {
                const dragEnd = { x: e.clientX, y: e.clientY };
                const delta = {
                    x: dragEnd.x - this.dragStart.x,
                    y: dragEnd.y - this.dragStart.y
                };
                this.dragStart = dragEnd;
                this.rotate(delta.x, delta.y);
            }
            e.preventDefault();
        });
        // Want to override mouse wheel to be zoom so handle those events on $img
        $img.addEventListener('wheel', (e) => {
            const delta = e.deltaY;
            this.zoom(delta);
            e.preventDefault();
        });
    }
    start() {
        this.shouldRun = true;
        this.renderLoop();
    }
    stop() {
        this.shouldRun = false;
    }
    zoom(delta) {
        const zoomSpeed = 0.005;
        const zoom = zoomSpeed * delta;
        let tx = this.cameraTransform[0][3];
        let ty = this.cameraTransform[1][3];
        let tz = this.cameraTransform[2][3];
        let dirx = this.cameraTransform[0][2];
        let diry = this.cameraTransform[1][2];
        let dirz = this.cameraTransform[2][2];
        tx += dirx * zoom;
        ty += diry * zoom;
        tz += dirz * zoom;
        this.cameraTransform[0][3] = tx;
        this.cameraTransform[1][3] = ty;
        this.cameraTransform[2][3] = tz;
    }
    rotate(deltaX, deltaY) {
        const rotationSpeed = 0.01;
        const rotation = {
            x: -rotationSpeed * deltaX,
            y: rotationSpeed * deltaY
        };
        const rx = rotation.x;
        const ry = rotation.y;
        const rz = 0;
        const rotationMatrix = [
            [Math.cos(rz) * Math.cos(ry), Math.cos(rz) * Math.sin(ry) * Math.sin(rx) - Math.sin(rz) * Math.cos(rx), Math.cos(rz) * Math.sin(ry) * Math.cos(rx) + Math.sin(rz) * Math.sin(rx), 0],
            [Math.sin(rz) * Math.cos(ry), Math.sin(rz) * Math.sin(ry) * Math.sin(rx) + Math.cos(rz) * Math.cos(rx), Math.sin(rz) * Math.sin(ry) * Math.cos(rx) - Math.cos(rz) * Math.sin(rx), 0],
            [-Math.sin(ry), Math.cos(ry) * Math.sin(rx), Math.cos(ry) * Math.cos(rx), 0],
            [0, 0, 0, 1]
        ];
        this.cameraTransform = this.matrixMultiply(this.cameraTransform, rotationMatrix);
    }
    matrixMultiply(a, b) {
        const aNumRows = a.length;
        const aNumCols = a[0].length;
        const bNumRows = b.length;
        const bNumCols = b[0].length;
        const m = new Array(aNumRows);
        for (let r = 0; r < aNumRows; ++r) {
            m[r] = new Array(bNumCols);
            for (let c = 0; c < bNumCols; ++c) {
                m[r][c] = 0;
                for (let i = 0; i < aNumCols; ++i) {
                    m[r][c] += a[r][i] * b[i][c];
                }
            }
        }
        return m;
    }
    async renderLoop() {
        while (this.shouldRun) {
            // render the image
            try {
                await this.fetchAndRenderImage();
                await delay(10);
            }
            catch (e) {
                console.error(e);
                this.shouldRun = false;
            }
        }
    }
    async fetchAndRenderImage($img) {
        const response = await fetch('/api/render', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                "cam_transform": this.cameraTransform
            })  
        });
    
        if (!response.ok) {
            throw new Error("HTTP error! status: ${response.status}");
        }
    
        const blob = await response.blob();
        const reader = new FileReader();
        reader.onloadend = () => {
            this.$img.src = reader.result;
        };
        if (blob) {
            reader.readAsDataURL(blob);
        }
    }
}
    
    

