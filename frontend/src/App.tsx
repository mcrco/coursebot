import './App.css'
import { ChatWindow } from './components/chat-window'
import { ThemeProvider } from "@/components/theme-provider"

function App() {
    return (
        <ThemeProvider defaultTheme="dark" storageKey="vite-ui-theme">
            <ChatWindow />
        </ThemeProvider>
    )
}

export default App
