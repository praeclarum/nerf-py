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
    }
    start() {
        this.shouldRun = true;
        this.renderLoop();
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
    
    

