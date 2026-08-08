package com.zenith.growthhub

import android.annotation.SuppressLint
import android.os.Bundle
import android.webkit.WebResourceRequest
import android.webkit.WebView
import android.webkit.WebViewClient
import androidx.appcompat.app.AppCompatActivity

class MainActivity : AppCompatActivity() {

    private lateinit var webView: WebView

    @SuppressLint("SetJavaScriptEnabled")
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState: Bundle?)
        
        // Initialize WebView
        webView = WebView(this)
        setContentView(webView)

        // Configure WebView parameters
        val settings = webView.settings
        settings.javaScriptEnabled = true
        settings.domStorageEnabled = true // Essential for localStorage (saves XP and daily progress)
        settings.databaseEnabled = true
        settings.allowFileAccess = true
        settings.useWideViewPort = true
        settings.loadWithOverviewMode = true
        
        // Force links to stay inside the WebView context rather than launching an external browser
        webView.webViewClient = object : WebViewClient() {
            override fun shouldOverrideUrlLoading(view: WebView?, request: WebResourceRequest?): Boolean {
                val url = request?.url?.toString() ?: ""
                if (url.startsWith("file://") || url.startsWith("http://localhost")) {
                    return false
                }
                return false
            }
        }

        // Load static web build packaged in assets
        webView.loadUrl("file:///android_asset/index.html")
    }

    // Capture device back clicks to navigate backwards inside the learning app tabs rather than exiting
    override fun onBackPressed() {
        if (webView.canGoBack()) {
            webView.goBack()
        } else {
            super.onBackPressed()
        }
    }
}
