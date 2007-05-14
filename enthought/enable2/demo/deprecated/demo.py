#-------------------------------------------------------------------------------
#
#  An Enable demo based upon the enVisage framework.
#
#  Written by: David C. Morrill
#
#  Date: 08/31/2004
#
#  (c) Copyright 2004 by Enthought, Inc.
#
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
#  Imports:
#-------------------------------------------------------------------------------

import wx

from canvas import create_enable_window, create_templates, RootNode, \
                   create_active_components, create_gridded_canvas_window

from enthought.envisage.application      import Application
from enthought.envisage.mdi_page_layout2 import MDIPageLayout2
from enthought.envisage.perspective      import Perspective
from enthought.envisage.explorer_view    import ExplorerView
from enthought.envisage.node             import Node
from enthought.envisage.node_tree_model  import NodeTreeModel
#from enthought.logging.log_view          import LogView
from enthought.logger.widget.logger_widget import LoggerWidget as LogView
# ui_hack
#from enthought.enable.enable_traits      import trait_sheet_handler

#import enthought.traits.wxtrait_sheet as trait_sheet

#-------------------------------------------------------------------------------
#  Constants:
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
#  'EnablePerspective' class:
#-------------------------------------------------------------------------------

class EnablePerspective ( Perspective ):

    #---------------------------------------------------------------------------
    #  Create the designer layout:
    #---------------------------------------------------------------------------

    def create_initial_layout ( self, layout ):
        """ Creates the initial perspective. """

        # Located here so we can test the app without triggering an import:
        from enthought.envisage.shell import Shell

        # hack
        self.scrollbar_dx = wx.SystemSettings.GetMetric( wx.SYS_VSCROLL_X )

        # Save the layout reference for later use:
        self.layout = layout

        application = self.application
        frame       = self.frame

        # Create the Python shell window:
        location    = layout.bottom.right
        frame.shell = Shell( location, -1, application, frame )
        location.AddPage( frame.shell, 'Python' )

        # Create the output window:
        location      = layout.bottom.right
        self.log_view = LogView( location )
        location.AddPage( self.log_view, 'Output' )

        # Create the 'templates' window:
        location = layout.left.top
        panel    = wx.Panel( location, -1 )
        sizer    = wx.BoxSizer( wx.VERTICAL )
        panel.SetAutoLayout( True )
        panel.SetSizer( sizer )

        node = RootNode( 'Templates', icon = 'images/templates.gif' )
        create_templates( node )
        self.templates = ExplorerView( panel, -1, application, frame,
                                       model = NodeTreeModel( node ) )
        sizer.Add( self.templates, 1, wx.EXPAND )
        sizer.Fit( panel )
        location.AddPage( panel, 'Templates' )

        # Create the simple 'Enable' window:
        create_enable_window( application, frame, layout.right.top )

        # Create the gridded canvas 'Enable' window:
#        grid = create_gridded_canvas_window( application, frame,
#                                             layout.right.top )
#        grid.on_trait_change( self.edit_component, 'component_context' )

        # Create the 'active components' window:
        self.components = create_active_components( application, frame,
                                                    layout.left.bottom )
        self.components.on_trait_change( self.component_selected, 'selection' )

        # ui_hack
        ## Create the current selected component trait sheet editor window:
        #location         = layout.bottom.left
        #self.trait_panel = tpanel = wx.ScrolledWindow( location, -1 )
        #location.AddPage( tpanel, 'Traits' )

    #---------------------------------------------------------------------------
    #  Edit the specified component:
    #---------------------------------------------------------------------------

#   def edit_component ( self, component ):
#       """ Edits the specified component.
#       """
#       try:
#           tpanel = self.trait_panel
#           tpanel.SetSizer( None )
#           tpanel.DestroyChildren()
#           sizer = wx.BoxSizer( wx.VERTICAL )
#           sheet = trait_sheet.TraitSheet( tpanel, component,
#                                           handler = trait_sheet_handler )
#           sizer.Add( sheet, 0, wx.EXPAND )
#           tpanel.SetAutoLayout( True )
#           tpanel.SetSizer( sizer )
#           tpanel.SetScrollRate( 0, 16 )
#           width, height = sheet.GetSize()
#           tpanel.SetSize( wx.Size( width + self.scrollbar_dx, height ) )
#           tpanel.GetParent().Layout()
#           self.layout.bottom.left.SetPageText( 0,
#               component.__class__.__name__ + ' Traits' )
#       except:
#           import traceback
#           traceback.print_exc()

    # ui_hack
    ##---------------------------------------------------------------------------
    ##  Handle a component being selected in the 'active components' window:
    ##---------------------------------------------------------------------------
    #
    #def component_selected ( self, selection ):
    #    if len( selection ) > 0:
    #       component = selection[0].data
    #       if component is not None:
    #           self.edit_component( component )

#-------------------------------------------------------------------------------
#  Main program:
#-------------------------------------------------------------------------------

Application( perspective     = EnablePerspective(),
             mdi_page_layout = MDIPageLayout2( style    = 'h',
                                               geometry = ( 0, 2, 2, 1 ) ),
             splash_image    = './images/splash.jpg',
             splash_timeout  = 10000,
             plugin_path     = ['']

).start()
