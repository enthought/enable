import time

from enthought.savage.svg.backends.kiva import renderer

class RenderPanel(renderer.Canvas):
    def __init__(self, parent, document=None):
        renderer.Canvas.__init__(self, parent)
        self.lastRender = None
        self.document = document
        self.zoom = 100
#        self.offset = wx.Point(0,0)


    def do_draw(self, gc):
        start = time.time()
        gc.clear()

        if not self.document:
            gc.show_text_at_point("No Document", 20, 20)
            return
        
        # SVG origin is upper right with y positive is down. argh.
        # Set up the transforms to fix this up.
        gc.translate_ctm(0, gc.height())
        # zoom percentage
        scale = float(self.zoom) / 100.0
        gc.scale_ctm(scale, -scale)
        self.document.render(gc)
        self.lastRender = time.time() - start
        print 'draw time:', self.lastRender
        
        
    def GetBestSize(self):
        if not self.document:
            return (-1,-1)
        sz = map(int,self.document.tree.getroot().get("viewBox").split())
        return wx.Rect(*sz).GetSize()
    
    def OnWheel(self, evt):
        self.zoom += (evt.m_wheelRotation / evt.m_wheelDelta) * 10
        self.Refresh()
    
    def OnLeftDown(self, evt):
        self.SetCursor(wx.StockCursor(wx.CURSOR_HAND))
        self.CaptureMouse()
        self.offsetFrom = evt.GetPosition()
        evt.Skip()
    
    def OnLeftUp(self, evt):
        if self.HasCapture():
            self.ReleaseMouse()
        self.SetCursor(wx.NullCursor)
        evt.Skip()
        
    def OnMotion(self, evt):
        if not self.HasCapture():
            return
        self.offset += (evt.GetPosition() - self.offsetFrom)
        self.offsetFrom = evt.GetPosition()
        self.Refresh()
        
    def OnMiddleClick(self, evt):
        self.offset = wx.Point(0,0)
        self.zoom = 100
        self.Refresh()
     
#class ViewFrame(wx.Frame):
#    #status bar cell locations
#    SCROLL_OFFSET = 0
#    FILE = 1
#    LOAD_TIME = 2
#    RENDER_TIME = 3
#    def __init__(self, parent):
#        wx.Frame.__init__(self, parent, style=wx.DEFAULT_FRAME_STYLE | wx.CLIP_CHILDREN)
#        self._mgr = wx.aui.AuiManager()
#        self._mgr.SetManagedWindow(self)
#        
#        self.wrap = wx.Panel(self)
#        self.profileLoading = True
#        
#        self.tree = XMLTree(self, None)
#        self.profileResults = ProfileResults(self)
#        self.render = RenderPanel(self.wrap)
#        self.reference = ReferencePanel(self.wrap, None)
#        sz = wx.BoxSizer(wx.HORIZONTAL)
#        sz.Add(self.render, 1, wx.EXPAND|wx.RIGHT, 1)
#        sz.Add(self.reference, 1, wx.EXPAND|wx.LEFT, 1)
#        self.wrap.SetSizer(sz)
#        
#        self.SetMenuBar(self.makeMenus())
#        self.SetToolBar(self.makeToolBar())
#        
#        self._mgr.AddPane(
#            self.tree, 
#                wx.aui.AuiPaneInfo().
#                    Top().
#                    CloseButton(False).
#                    Layer(1).
#                    Caption("XML Tree").
#                    MinSize(self.tree.GetBestSize()),
#            "XML TREE"
#        )
#        self._mgr.AddPane(
#            self.profileResults, 
#                wx.aui.AuiPaneInfo().
#                    Top().
#                    CloseButton(False).
#                    Layer(1).
#                    Caption("PROFILE RESULTS").
#                    MinSize(self.tree.GetBestSize()),
#            "PROFILE RESULTS"
#        )
#        
#        self._mgr.AddPane(
#            self.wrap, 
#            wx.aui.AuiPaneInfo().CentrePane().Caption("SVG Rendering"),
#            "VIEWER"
#        )        
#        self.CreateStatusBar(5)
#        #self.Maximize()
#        # THis is a nice size for comparing most images
#        self.SetSize((1000,600))
#        self._mgr.Update()
#        
#        self.Bind(wx.EVT_MENU, self.OnOpenFile, id=wx.ID_OPEN)
#        def OnProfileLoading(evt):
#            self.profileLoading = bool(evt.Checked())
#        self.Bind(wx.EVT_MENU, OnProfileLoading, id=wx.ID_FORWARD)
#        self.Bind(wx.EVT_MENU, self.Reload, id=wx.ID_REFRESH)
#        self.Bind(wx.EVT_MENU, lambda x:self.Destroy(), id=wx.ID_EXIT)
#        self.Bind(wx.EVT_CHOICE, self.OnChooseFile)
#        self.Bind(wx.EVT_TREE_SEL_CHANGED, self.OnTreeSelectionChange)
#        self.Bind(wx.EVT_UPDATE_UI, self.OnUpdateUI)
#        
#        
#        self.filePicker.SetSelection(self.filePicker.FindString('shapes-rect-01-t'))
#        self.OnChooseFile(None)
#        
#                
#    def makeMenus(self):
#        fileMenu = wx.Menu()
#        mi = wx.MenuItem(fileMenu, wx.ID_FORWARD, "Profile loading", kind=wx.ITEM_CHECK)
#        fileMenu.AppendItem(mi)
#        fileMenu.Append(wx.ID_OPEN, "&Open")
#        fileMenu.Append(wx.ID_REFRESH, "&Reload Current File\tF5")
#        fileMenu.AppendSeparator()
#        fileMenu.Append(wx.ID_EXIT, "E&xit")
#        
#        
#        mb = wx.MenuBar()
#        mb.Append(fileMenu, "&File")
#        
#        return mb
#
#    def makeToolBar(self):
#        tb = wx.ToolBar(self, style=wx.TB_FLAT)
#        self.filePicker = wx.Choice(tb, choices=self.getFileList())
#        tb.AddControl(self.filePicker)
#        tb.Realize()
#        return tb
#    
#    def getFileList(self):
#        #look for the test files in the "compliance" dir next to our package
#        files = os.listdir(self.getSVGDir())
#        splitted = map(os.path.splitext, files)
#        
#        return sorted(fname for fname, ext in splitted)
#    
#    def getSVGDir(self):
#        dir = os.path.dirname(document.__file__)
#        dir = os.path.join(dir, "compliance", "svg")
#        return dir
#        
#    def getPNGDir(self):
#        dir = os.path.dirname(document.__file__)
#        dir = os.path.join(dir, "compliance", "png")
#        return dir
#            
#    def Reload(self, evt):
#        self.openFile(self.currentFile)        
#    
#    def openFile(self, filenameOrBuffer):
#        start = time.time()
#        tree = etree.parse(filenameOrBuffer)
#        try:
#            if self.profileLoading:
#                import cProfile
#                p = cProfile.Profile()
#                p.enable()
#            if isinstance(filenameOrBuffer, basestring):
#                resources = document.ResourceGetter.fromfilename(
#                    filenameOrBuffer)
#            else:
#                resources = document.ResourceGetter()
#            self.document = document.SVGDocument(tree.getroot(),
#                resources=resources)
#            if self.profileLoading:
#                import pstats
#                p.disable()
#                results = pstats.Stats(p)
#                self.profileResults.SetResults(results)
#            else:
#                self.profileResults.SetResults(None)
#                
#            self.render.document = self.document
#        except:
#            #pdb.set_trace()
#            import traceback
#            self.render.document = None
#            traceback.print_exc()
#        
#        amount = time.time() - start
#        self.tree.updateTree(tree)
#        self.SetStatusText("Loaded in %2f seconds" % amount, self.LOAD_TIME)
#        self.SetStatusText(filenameOrBuffer)
#        self.currentFile = filenameOrBuffer
#        
#        self.render.dirty = 1
#        self.Refresh()
#    
#       
#    def OnChooseFile(self, evt):
#        fname = self.filePicker.GetString(self.filePicker.GetSelection())
#        if fname == '':
#            return
#        svg = os.path.join(self.getSVGDir(), fname+'.svg')
#        self.openFile(svg)
#        for prefix in ['full-', '']:
#            png = os.path.join(self.getPNGDir(), prefix+fname+'.png')
#            if os.path.isfile(png):
#                self.reference.bmp = wx.Bitmap(png)
#                break
#        else:
#            self.reference.bmp = None
#        
#    def OnOpenFile(self, evt):
#        dlg = wx.FileDialog(self)
#        if dlg.ShowModal() == wx.ID_OK:
#            self.openFile(dlg.GetPath())
#            self.reference.bmp = None
#        
#    def OnTreeSelectionChange(self, evt):
#        item = self.tree.GetSelection()
#        element = self.tree.GetItemPyData(item)
#        if element is None:
#            return
#        path = self.document.paths[element]
#        print path
#        
#    def OnUpdateUI(self, evt):
#        if self.render.lastRender is not None:
#            self.SetStatusText("Rendered in %2f seconds" % self.render.lastRender, self.RENDER_TIME)
#        if evt.Id == wx.ID_FORWARD:
#            evt.Checked = self.profileLoading
