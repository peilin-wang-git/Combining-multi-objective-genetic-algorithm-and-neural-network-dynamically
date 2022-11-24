'#Language "WWB-COM"

Option Explicit
Public cstProjectPath As String
Public paramFileDstPath As String
Sub Main
    Dim numofparams As Long
    Dim N As Long
    Dim parname As String
    Dim svalue As String
    Dim description As String
    cstProjectPath = "%CSTPROJFILE%"
    paramFileDstPath = "%PARAMDSTPATH%"
    OpenFile(cstProjectPath)
    numofparams=GetNumberOfParameters()

    Open(paramFileDstPath) For Output As #1
    Print #1,paramFileDstPath
	Print #1,cstProjectPath
    Print #1,numofparams
    For N=0 To numofparams-1
        parname=GetParameterName (N)
        svalue=GetParameterSValue(N)
        description=GetParameterDescription (parname)
        Print #1,N;vbTab;parname;vbTab;svalue;vbTab;description
    Next
	Close #1
    Quit
End Sub