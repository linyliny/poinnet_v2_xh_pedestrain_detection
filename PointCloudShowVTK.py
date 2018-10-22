import numpy as np
import vtk


def show(data, color):
        
    # Point size
    point_size = 1
        
    # Create the geometry of a point (the coordinate)
    points = vtk.vtkPoints()
    # Create the topology of the point (a vertex)
    vertices = vtk.vtkCellArray()
    # Setup colors
    Colors = vtk.vtkUnsignedCharArray()
    Colors.SetNumberOfComponents(3)
    Colors.SetName("Colors")
    
    # Add points
    for i in range(0, len(data)):
        point = data[i]
        r, g, b = color[i]
        id = points.InsertNextPoint(point)
        vertices.InsertNextCell(1)
        vertices.InsertCellPoint(id)
        Colors.InsertNextTuple3(r, g, b)
   
    # Create a polydata object
    point = vtk.vtkPolyData()
    # Set the points and vertices we created as the geometry and topology of the polydata
    point.SetPoints(points)
    point.SetVerts(vertices)
    point.GetPointData().SetScalars(Colors)
    point.Modified()
    # Visualize
    mapper = vtk.vtkPolyDataMapper()
    if vtk.VTK_MAJOR_VERSION <= 5:
        mapper.SetInput(point)
    else:
        mapper.SetInputData(point)
    
    ## ACTOR
    # Create an actor
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetPointSize(point_size)
    axes = vtk.vtkAxesActor()
    
    
    ## RENDER
    renderer = vtk.vtkRenderer()
    # Add actor to the scene
    renderer.AddActor(actor)
    # renderer.AddActor(axes)
    # Background
    renderer.SetBackground(0.1, 0.2, 0.3)
    # Reset camera
    renderer.ResetCamera()
    # Render window
    renderWindow = vtk.vtkRenderWindow()
    renderWindow.AddRenderer(renderer)
    # Interactor
    renderWindowInteractor = vtk.vtkRenderWindowInteractor()
    renderWindowInteractor.SetRenderWindow(renderWindow)
    # Begin interaction
    renderWindow.Render()
    renderWindowInteractor.Start()