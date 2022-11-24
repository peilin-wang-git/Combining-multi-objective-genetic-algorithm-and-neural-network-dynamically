'#Language "WWB-COM"

Option Explicit

Public cstProjectPath As String
Public taskFileDir As String
Public resultDir As String
Public index As Integer
Public configFullPath,outFullDir As String
Public currentResultName As String
Public cstType As String
Public paramName() As String
Public paramValue() As String
Public totalElaspedTime As Double
Public postProcessTime As Double
Sub Main
	Dim cp As String
	cp="Ver 20200811"
	configFullPath="%CONFIGFILEPATH%"
	LoadConfig(configFullPath)
	index=0
	Dim maxAttempt,count As Long
	maxAttempt=999999
	count=0
	OpenFile(cstProjectPath)
	'EXTERN'prepareEnv()
	'WORKING LOOP
	Dim startTime,endTime As Double
	Dim status As String
	Do While count < maxAttempt
		status=GetJobDef(index)
		If status = "NewTask" Then
			
			CreateResultDir()
			If not IsFileExists(taskFileDir & index &".success") Then 
				startTime=Timer
				ChangeParams()
				'EXTERN'Start()
				endTime=Timer
				totalElaspedTime=endTime-startTime
				Success()
				
			End If
			count=0
			index +=1
			
				

		ElseIf status = "Idle" Then

			count += 1

		ElseIf status = "Terminate" Then

			Exit Do

		ElseIf status = "Error" Then

			Exit Do


		Else
			Exit Do

		End If

		Wait 1
	Loop

	SaveAs (taskFileDir &"temp.cst", False)
	Quit

End Sub

Function CreateResultDir()
	If Not IsFileExists(resultDir & currentResultName) Then 
		MkDir (resultDir & currentResultName)
	End If

End Function
Function LoadConfig(confPath As String) As String
	
	Open (confPath) For Input As #1
	'GetCSTType
	Line Input #1, cstType

	'GetResultDir
	Line Input #1, resultDir
	If Right(resultDir,1)<>"\" Then
		resultDir=resultDir & "\"
	End If
	'GetCstProjPath
	Line Input #1, cstProjectPath
	
	'GetTaskFileDir
	Line Input #1,taskFileDir
	If Right(taskFileDir,1)<>"\" Then
		taskFileDir=taskFileDir & "\"
	End If

	Close #1
End Function

Function Success()
	Open(taskFileDir & index &".success") For Output As #1
	Print #1,currentResultName
	Print #1,
	Print #1,"success"
	Close #1
	Dim i As Integer
	Open(resultDir & currentResultName &"\run.log") For Output As #1
	For i = LBound(paramName) To UBound(paramName)
		Print #1,paramName(i)
		Print #1,paramValue(i)
		Print #1,
	Next i
	Print #1,"totalElaspedTime"
	Print #1,totalElaspedTime
	Print #1,"postProcessTime"
	Print #1,postProcessTime
	Close #1
End Function

Function IsFileExists(ByVal strFileName As String) As Boolean
    If Dir(strFileName, 16) <> Empty Then
        IsFileExists = True
    Else
        IsFileExists = False
    End If
End Function

Function GetJobDef(index As Integer) As String
	If IsFileExists(taskFileDir & "terminate.txt") Then
		GetJobDef="Terminate"
		Exit Function
	End If
	GetJobDef="Idle"
	Debug.Print taskFileDir & index &".txt"
	If Not IsFileExists(taskFileDir & index &".txt") Then
		Exit Function
	End If
	
	Dim strTemp As String
	Dim n,i As Integer
	n=0
	i=0
	wait 0.1
	Open(taskFileDir & index &".txt") For Input As #1
	Line Input #1,currentResultName
	Do While Not EOF(1)
		Line Input #1, strTemp
		Debug.Print strTemp
		n=n+1
	Loop
	Close #1
	If n=0 Then
		Exit Function
	End If
	ReDim paramName(1 To n/2)
	ReDim paramValue(1 To n/2)
	Open(taskFileDir & index &".txt") For Input As #1
	Line Input #1,currentResultName
	Do While Not EOF(1)
		i=i+1
		Line Input #1, paramName(i)
		Line Input #1, paramValue(i)
	Loop
	Close #1
	GetJobDef="NewTask"
End Function

Sub ChangeParams
	Dim i As Integer
	For i = LBound(paramName) To UBound(paramName)
		StoreDoubleParameter(paramName(i),paramValue(i))
	Next i
	Rebuild
End Sub
