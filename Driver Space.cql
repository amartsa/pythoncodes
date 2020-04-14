/// Section 1.1 Add constraints to ensure unique IDs on Numbers. Accident nodes can have the same Report Number, but NOT same Sq ID.

CREATE CONSTRAINT ON (accid:Accident) ASSERT accid.seq IS UNIQUE;
CREATE CONSTRAINT ON (accid:Accident) ASSERT (accid.reportNumber, accid.seq) IS NODE KEY;
CREATE CONSTRAINT ON (int:Intersections) ASSERT int.IntersectionID IS UNIQUE;


//// Section 1.2 We load and create a node for each Accident instance, with a unique Sq ID, although Report Numbers can be repeated (people of different ages involved in same accident for example)
LOAD CSV WITH HEADERS FROM 'file:///Dataset.csv' AS colName
MERGE (:Accident {reportNumber: colName.ReportNumber, Hour: toInteger(colName.CRASH_HOD), Day: colName.CRASH_DT, DOW: colName.CRASH_DOW,
		WOY: toInteger(colName.CRASH_WK), Month: toInteger(colName.CRASH_MOY), Year: toInteger(colName.CRASH_YEAR), 
		age: colName.Age, fatalities: colName.Nfatalities, injuries: colName.Ninjuries, alc: colName.AlcoholRel, dist: colName.DistractRel, 
		weath: colName.WeatherAtCrash, stName: colName.StreetName, intName: colName.IntName, seq: colName.Sq});


////Section 1.3 Here we are creating a list of nodes with the unique values from EACH Column.

// Age 
LOAD CSV WITH HEADERS FROM 'file:///Dataset.csv' AS colName
WITH collect(distinct toInteger(colName.Age)) as nodes
UNWIND nodes AS nodelist
MERGE (:Age {age: nodelist});

//Alcohol
LOAD CSV WITH HEADERS FROM 'file:///Dataset.csv' AS colName
WITH collect(distinct colName.AlcoholRel) as nodes
UNWIND nodes AS nodelist
MERGE (:AlcoholRel {was: nodelist});

//Distraction
LOAD CSV WITH HEADERS FROM 'file:///Dataset.csv' AS colName
WITH collect(distinct colName.DistractRel) as nodes
UNWIND nodes AS nodelist
MERGE (:DistractRel {was: nodelist});

//Weather
LOAD CSV WITH HEADERS FROM 'file:///Dataset.csv' AS colName
WITH collect(distinct colName.WeatherAtCrash) as nodes
UNWIND nodes AS nodelist
MERGE (:Weather {was: nodelist});

//Number of Fatalities
LOAD CSV WITH HEADERS FROM 'file:///Dataset.csv' AS colName
WITH collect(distinct toInteger(colName.Nfatalities)) as nodes
UNWIND nodes AS nodelist
MERGE (:Nfatalities {number: nodelist});

//Number of Injuries
LOAD CSV WITH HEADERS FROM 'file:///Dataset.csv' AS colName
WITH collect(distinct toInteger(colName.Ninjuries)) as nodes
UNWIND nodes AS nodelist
MERGE (:Ninjuries {number: nodelist});

//Hour of Accident
LOAD CSV WITH HEADERS FROM 'file:///Dataset.csv' AS colName
WITH collect(distinct toInteger(colName.CRASH_HOD)) as nodes
UNWIND nodes AS nodelist
MERGE (:Hour {hour: nodelist});

//Day of Week
LOAD CSV WITH HEADERS FROM 'file:///Dataset.csv' AS colName
WITH collect(distinct colName.CRASH_DOW) as nodes
UNWIND nodes AS nodelist
MERGE (:DayOfWeek {number: nodelist});

//Week of Year
LOAD CSV WITH HEADERS FROM 'file:///Dataset.csv' AS colName
WITH collect(distinct toInteger(colName.CRASH_WK)) as nodes
UNWIND nodes AS nodelist
MERGE (:WeekOfYear {number: nodelist});

//connect Accident and streets
MATCH (a:Accident), (st:Streets) WHERE a.stName = st.StreetName
MERGE (a)-[:LOCATED_AT]->(st);

//connect Accident and intersection
MATCH (a:Accident), (int:Intersections) where toString(a.intName) = toString(int.IntName)
MERGE (a)-[:LOCATED_AT]->(int);



////Section 1.4 Connecting the DRIVER SPACE nodes

// alcohol node
MATCH (a:Accident), (alc:AlcoholRel)
WHERE a.alc = alc.was 
MERGE (a)-[:WAS_DUI]->(alc);

//distraction node
MATCH (a:Accident), (dis:DistractRel)
WHERE a.dist = dis.was 
MERGE (a)-[:DISTRACTED]->(dis);

//age node
MATCH (a:Accident), (age:Age)
WHERE toInteger(a.age) = toInteger(age.age)
MERGE (a)-[:DRIVERAGE]->(age);


//// Section 1.5 Connecting the CONTEXT SPACE nodes

//fatalities node
MATCH (a:Accident), (fat:Nfatalities)
WHERE toInteger(a.fatalities) = toInteger(fat.number)
MERGE (a)-[:NUM_DEAD]->(fat);

//weather node
MATCH (a:Accident), (weat:Weather)
WHERE a.weath = weat.was
MERGE (a)-[:WEATHER]->(weat);

//injuries node
MATCH (a:Accident), (inj:Ninjuries)
WHERE toInteger(a.injuries) = toInteger(inj.number)
MERGE (a)-[:NUM_INJURED]->(inj);

//day of week node
MATCH (a:Accident), (dow:DayOfWeek)
WHERE a.DOW = dow.number
MERGE (a)-[:DAY_OF_WEEK]->(dow);

//week of year node
MATCH (a:Accident), (woy:WeekOfYear)
WHERE toInteger(a.WOY) = toInteger(woy.number)
MERGE (a)-[:WEEK_OF_YEAR]->(woy);

//hour of day node
MATCH (a:Accident), (hr:Hour)
WHERE toInteger(a.Hour) = toInteger(hr.hour)
MERGE (a)-[:HOUR_OF_DAY]->(hr);

