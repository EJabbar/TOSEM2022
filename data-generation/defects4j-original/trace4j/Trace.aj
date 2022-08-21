import org.aspectj.lang.Signature;
import org.aspectj.lang.reflect.MethodSignature;
import org.junit.Test;

import java.io.FileNotFoundException;
import java.io.PrintWriter;
import java.util.*;

public aspect Trace {

    private final PrintWriter writer = new PrintWriter("log.csv");
    private final PrintWriter writer_return = new PrintWriter("ret_log.csv");

    long beforeProceed = 0;
    long afterProceed = 0;
    private static final String START_PARAM = "<SP>";
    private static final String END_PARAM = "<EP>";
    private static final String NEXT_PARAM = "<NP>";

    Trace() throws FileNotFoundException {
        initLogFile();
    }

    pointcut traceMethods(): call(* * (..)) && !within(Trace) && !within(org.junit.internal..*) && !within(junit.framework.TestSuite) && !within(org.junit.runner*..*) && !within(org.junit.validator..*) && !within(com.sun.proxy..*);

    Object around(): traceMethods(){
        Signature target = thisJoinPointStaticPart.getSignature();
        Signature source = thisEnclosingJoinPointStaticPart.getSignature();
        Object[] objects = thisJoinPoint.getArgs();
        String returnType = ((MethodSignature) thisJoinPoint.getSignature()).getReturnType().getSimpleName();
        String arguments = argumentsToString(objects);
        long time = System.nanoTime();
        TraceLog traceLog = new TraceLog(time, source.getDeclaringTypeName(), source.getName(), target.getDeclaringTypeName(), target.getName(), arguments);
        writeLogWithoutReturn(traceLog);
        beforeProceed++;
        Object r = proceed();
        afterProceed++;
        String returnValue = returnValueToString(r);
        if (returnType.equals("void")){
            returnValue = "empty";
        }
        writeReturnValue(time, returnType, returnValue);
        return r;
    }

    pointcut testJUnit4xIsAboutToBegin(): (@annotation(Test));
    pointcut afterTestFinished(): call (* java.lang.System.exit(*));

    before(): afterTestFinished(){
        System.out.println("Before Log: " + beforeProceed);
        System.out.println("After Log: " + afterProceed);
    }

    private void initLogFile() {
        String sb = "id,calling_class,calling_method,called_class,called_method,called_method_args" + "\n";
        writer.write(sb);
        writer.flush();
        sb = "id,return_type,return_value" + "\n";
        writer_return.write(sb);
        writer_return.flush();
    }

    private String argumentsToString(Object[] objects) {
        StringBuilder string = new StringBuilder();
        string.append(START_PARAM);
        for (Object object : objects) {
            if (object == null) {
                String type = "object";
                String values = "null";
                string.append(type).append(":").append(values).append(NEXT_PARAM);
                continue;
            }
            String type = object.getClass().getSimpleName();
            String values = deepToString(new Object[]{object});
            if (values.contains("[")) {
                values = values.substring(1, values.length() - 1);
                //TODO: check with the null or empty string to find if the above code works
            }
            string.append(type).append(":").append(values);
            string.append(NEXT_PARAM);
        }
        if (string.lastIndexOf(NEXT_PARAM) == string.length() - NEXT_PARAM.length()) {
            string = new StringBuilder(string.substring(0, string.length() - NEXT_PARAM.length()));
        }
        string.append(END_PARAM);
        return string.toString();
    }

    private String returnValueToString(Object object) {
        StringBuilder string = new StringBuilder();
        if (object == null) {
            return "null";
        }
        String values = deepToString(new Object[]{object});
        if (values.contains("[")) {
            values = values.substring(1, values.length() - 1);
            //check with the null or empty string to find if the above code works
        }
        string.append(values);
        return string.toString();
    }

    private void writeLogWithoutReturn(TraceLog traceLog) {
        StringBuilder sb = new StringBuilder();
        sb.append('"'+traceLog.id+'"');
        sb.append(",");
        sb.append('"'+traceLog.callingClassName+'"');
        sb.append(",");
        sb.append('"'+traceLog.callingMethodName+'"');
        sb.append(",");
        sb.append('"'+traceLog.calledClassName+'"');
        sb.append(",");
        sb.append('"'+traceLog.calledMethodName+'"');
        sb.append(",");
        sb.append('"'+traceLog.args+'"');
        sb.append("\n");
        writer.write(sb.toString());
        writer.flush();
    }

    private void writeReturnValue(long id, String returnType, String returnValue) {
        StringBuilder sb = new StringBuilder();
        sb.append('"'+id+'"');
        sb.append(",");
        sb.append('"'+returnType+'"');
        sb.append(",");
        sb.append('"'+returnValue+'"');
        sb.append("\n");
        writer_return.write(sb.toString());
        writer_return.flush();
    }

    public static String deepToString(Object[] v) {
        if (v == null)
            return "null";
        HashSet<Object[]> seen = new HashSet<>();
        StringBuilder b = new StringBuilder();
        deepToString(v, b, seen);
        return b.toString();
    }

    private static void deepToString(Object[] v, StringBuilder b, HashSet<Object[]> seen) {
        b.append("[");
        for (int i = 0; i < v.length; ++i) {
            if (i > 0)
                b.append(", ");
            Object elt = v[i];
            if (elt == null)
                b.append("null");
            else if (elt instanceof boolean[])
                b.append(Arrays.toString((boolean[]) elt));
            else if (elt instanceof byte[])
                b.append(Arrays.toString((byte[]) elt));
            else if (elt instanceof char[])
                b.append(Arrays.toString((char[]) elt));
            else if (elt instanceof short[])
                b.append(Arrays.toString((short[]) elt));
            else if (elt instanceof int[])
                b.append(Arrays.toString((int[]) elt));
            else if (elt instanceof long[])
                b.append(Arrays.toString((long[]) elt));
            else if (elt instanceof float[])
                b.append(Arrays.toString((float[]) elt));
            else if (elt instanceof double[])
                b.append(Arrays.toString((double[]) elt));
            else if (elt instanceof Object[]) {
                Object[] os = (Object[]) elt;
                if (seen.contains((Object[]) elt))
                    b.append("[...]");
                else {
                    seen.add((Object[]) elt);
                    deepToString(os, b, seen);
                }
            } else if (elt.getClass().getName().startsWith("java.lang") || elt instanceof String)
                b.append(elt);
            else
                b.append(elt.getClass().getSimpleName() + "@" + Integer.toHexString(System.identityHashCode(elt)));
        }
        b.append("]");
    }

    public static class TraceLog {
        public long id;
        public String callingClassName;
        public String callingMethodName;
        public String calledClassName;
        public String calledMethodName;
        public String args;

        TraceLog(long id, String callingClassName, String callingMethodName, String calledClassName, String calledMethodName, String args) {
            this.id = id;
            this.callingClassName = callingClassName;
            this.callingMethodName = callingMethodName;
            this.calledClassName = calledClassName;
            this.calledMethodName = calledMethodName;
            this.args = args;
        }
    }
}