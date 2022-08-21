# Trace4j

Trace4j can be used for logging the traces of running tests of Java applications by using AspectJ. 
Currently:

   - Defects4j repository is supported.
   - All testable classes of one project is run and traces are logged.
   - calling_class, calling_method, called_class, called_method, called_method_args, return_type, return_value are logged.
   
Other projects or use cases can be integrated easily. **Your PRs are welcome!**

## How to run?

0. Install Java 1.8 and [Defects4j](https://github.com/rjust/defects4j)

1. Clone the repository
    - ```git clone https://github.com/EhsanMashhadi/trace4j.git```
2. Install required dependencies
    - ```bash setup_env.sh ```
3. Logging the traces
    - ```bash run.sh Codec 1b```

## What is the structure?
It creates two files for each CUT (class under test) in the ```./projects/name_version/trace4j/logs/```:
1. `log_[CUT].csv`
2. `ret_log_[CUT].csv`

### called_method_args structure?
Each method arguments is shown as `Type:Value`. You can check the `Trace.aj` file for the complete logic.

There are some delimiters in called_method_args columns:

`<SP>`: start of parameters

`<NP>`: delimiter between parameters
   
`<EP>`: end of parameters


### Structure of `log_[CUT].csv`

|id|calling_class|calling_method|called_class|called_method|called_method_args|
|---|---|---|----|---|---|
|`4112177068950`|`"org.apache.commons.codec.language.Caverphone"`|`"caverphone"`|`"java.lang.String"`|`"substring"`|`"<SP>Integer:0<NP>Integer:10<EP>"`|
|`4112177094891`|`"org.apache.commons.codec.language.CaverphoneTest"`|`"testSpecificationExamples"`|`"junit.framework.TestCase"`|`"assertEquals"`|`"<SP>String:PTA1111111<NP>String:PTA1111111<EP>"`|
|`4112177108025`|`"junit.framework.TestCase"`|`"assertEquals"`|`"junit.framework.Assert"`|`"assertEquals"`|`"<SP>String:PTA1111111<NP>String:PTA1111111<EP>"`|


### Structure of `ret_log_[CUT].csv`

|id|return_type|return_value|
|---|---|---|
|`4112177068950`|`"String"`|`"PTA1111111"`|
|`4112177094891`|`"void"`|`"empty"`|
|`4112177108025`|`"void"`|`"empty"`|

You can find the return type and return value of each method by joining two files `log_[CUT].csv` and `ret_log_[CUT].csv` on the id columns.


## Licensing
It is licensed under the MIT license. See the [LICENSE.md](https://github.com/EhsanMashhadi/trace4j/blob/master/LICENSE) for the complete license.