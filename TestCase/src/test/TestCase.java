package test;
import java.util.*;
import org.apache.commons.codec.digest.*;
public class TestCase {
	public TestCase(){
		
	}
	public static void main(String[] arg){
		/*
		int A[] = {1};
		int B[] = {2};
		merge(A,1,B,1);
		*/
		
		TestCase myTestCase = new TestCase();
		/*
		ListNode a = null;
		ListNode b = myTestCase.new ListNode(0);
		myTestCase.mergeTwoLists(a, b);
		*/
		/*
		String a = "a";
		String b = "a";
		myTestCase.strStr(a,b);
		myTestCase.solveNQueens(8);
		*/
		/*
		int[] input = {-2,-1,-3,4,-1,2,1,-5,4};
		System.out.println(myTestCase.maxSubArray(input));
		*/
		/*
		int[] testSet = {3,7,11,13,17};
		ArrayList<Integer> prefix = new ArrayList<Integer>();
		ArrayList<ArrayList<Integer>> result = new ArrayList<ArrayList<Integer>>();
		myTestCase.printSelectionWays(testSet, 21, prefix, result);
		for(int i=0; i<result.size();++i){
			for(int j=0;j<result.get(i).size();++j){
				System.out.print(result.get(i).get(j) + " ");
			}
			System.out.print("\n");
		}
		*/
		/*
		int[] testSet = {3,7,11,13,17};
		System.out.println(myTestCase.numberOfWays(testSet, 5, 4));
		*/
		/*
		ArrayList<Interval> myTestIntervals = new ArrayList<Interval>();
		myTestIntervals.add(myTestCase.new Interval(1,3));
		myTestIntervals.add(myTestCase.new Interval(6,9));
		Interval newInterval = myTestCase.new Interval(2,5);
		myTestCase.insert(myTestIntervals, newInterval);
		TreeNode test= myTestCase.new TreeNode(1);
		myTestCase.isValidBST(test);
		*/
		/*
		int[] A= {2,0,6,9,8,4,5,0,8,9,1,2,9,6,8,8,0,6,3,1,2,2,1,2,6,5,3,1,2,2,6,4,2,4,3,0,0,0,3,8,2,4,0,1,2,0,1,4,6,5,8,0,7,9,3,4,6,6,5,8,9,3,4,3,7,0,4,9,0,9,8,4,3,0,7,7,1,9,1,9,4,9,0,1,9,5,7,7,1,5,8,2,8,2,6,8,2,2,7,5,1,7,9,6};
		System.out.print(myTestCase.canJump(A));
		
		ListNode head = null;
		myTestCase.deleteDuplicates(head);
		
		String start = "hit";
		String end = "cog";
		HashSet<String> dict = new HashSet<String>(Arrays.asList("hot","dot","dog","lot","log"));
		System.out.print(myTestCase.ladderLength(start, end, dict));
		*/
		/*
		System.out.print(myTestCase.isPalindrome(1001));
		
		System.out.print(myTestCase.longestPalindrome("ccc"));
		*/
		/*
		int[] num = {0,1};
		myTestCase.permute(num);
		*/
		/*
		System.out.print(myTestCase.decodeWays("34125"));
		*/
		/*
		int[] num = {5,4,1,2};
		System.out.print(myTestCase.trap(num));
		*/
		/*
		System.out.print(myTestCase.addBinary("0", "0"));
		*/
		/*
		TreeNode tn = myTestCase.new TreeNode(-3);
		System.out.print(myTestCase.maxPathSum(tn));
		*/
		/*
		ListNode ln = myTestCase.new ListNode(3);
		ln.next = myTestCase.new ListNode(4);
		ListNode travel = ln.next;
		travel.next = myTestCase.new ListNode(5);
		travel = travel.next;
		travel.next = myTestCase.new ListNode(6);
		travel = travel.next;
		travel.next = myTestCase.new ListNode(7);
		myTestCase.sortedListToBST(ln);
		*/
		//System.out.print(myTestCase.divide(-2147483648, 1));
		/*
		int[] a = new int[] {9};
		myTestCase.plusOne(a);
		*/
		/*
		LruCache testCache = new LruCache(3);
		testCache.put(3, "3");
		testCache.put(4, "4");
		testCache.put(5, "5");
		testCache.put(6, "6");
		testCache.get(4);
		testCache.get(5);
		testCache.put(7, "7");
		testCache.printLinkedList();
		*/
		/*
		String input = "abcdefg";
		System.out.println(DigestUtils.md5Hex(input));
		*/
		/*
		System.out.print(myTestCase.totalNQueens(6));
		*/
		/*
		Set<String> dict = new HashSet<String>();
		dict.add("cat");
		dict.add("cats");
		dict.add("sand");
		dict.add("dog");
		myTestCase.wordBreak("catsanddog", dict);
		*/
		//System.out.print(myTestCase.minCut("abbab"));
		/*
		int[] input = new int[]{3,10,24,-24,12,9,7,15,-3};
		int[] mergeResult = myTestCase.mergeSort(input);
		for(int i=0;i<mergeResult.length;++i){
			System.out.print(mergeResult[i]);
			System.out.print(' ');
		}
		*/
		System.out.print(myTestCase.minWindow("ab", "b"));
	}
	public void merge(int A[], int m, int B[], int n) {
        // Start typing your Java solution below
        // DO NOT write main() function
        while(n > 0){
            if(m<=0 || A[m-1] < B[n-1]){
                A[m+n-1] = B[n-1];
                n--;
            }else{
                A[m+n-1] = A[m-1];
                m--;
            }
        }
    }
	
	/*
	 * leetcode editor distance
	 * @word1, @word2 2 input word string
	 * @return minimum editor distance between word1 and word2
	 */
	
    public int minDistance(String word1, String word2) {
        // Start typing your Java solution below
        // DO NOT write main() function
        int word1Len = word1.length();
        int word2Len = word2.length();
        if(word1Len == 0 || word2Len == 0){
            return word1Len > word2Len ? word1Len : word2Len;
        }
        int[][] minDistanceTrack = new int[word1Len+1][word2Len+1];
        for(int i=0; i<=word1Len; ++i){
            minDistanceTrack[i][0] = i;
        }
        for(int i=1; i<=word2Len; ++i){
            minDistanceTrack[0][i] = i;
        }
        for(int i=1;i<=word1Len;++i){
            for(int j=1;j<=word2Len;++j){
                if(word1.charAt(i-1) == word2.charAt(j-1)){
                    minDistanceTrack[i][j] = minDistanceTrack[i-1][j-1];
                }else{
                    minDistanceTrack[i][j] = Math.min(minDistanceTrack[i-1][j-1],Math.min(minDistanceTrack[i-1][j],minDistanceTrack[i][j-1])) + 1;
                }
            }
        }
        return minDistanceTrack[word1Len][word2Len];
    }
	
    
    /*
     * leetcode 3 sum
     */
    public ArrayList<ArrayList<Integer>> threeSum(int[] num) {
        ArrayList<ArrayList<Integer>> result = new ArrayList<ArrayList<Integer>>();
        int inputLength = num.length;
        
        HashMap<ArrayList<Integer>,Integer> checkDupMap = new HashMap<ArrayList<Integer>,Integer>();
        Arrays.sort(num);
        for(int currentIndex = 0; currentIndex<inputLength-2; ++currentIndex){
            int lowIndex = currentIndex+1;
            int highIndex = inputLength-1;
            while(lowIndex<highIndex){
                if(num[lowIndex] + num[currentIndex] + num[highIndex] == 0){
                    ArrayList<Integer> oneEntry = new ArrayList<Integer>();
                    oneEntry.add(num[currentIndex]);
                    oneEntry.add(num[lowIndex]);
                    oneEntry.add(num[highIndex]);
                    if(checkDupMap.get(oneEntry) == null){
                        checkDupMap.put(oneEntry,1);
                        result.add(oneEntry);
                    }
                    lowIndex++;
                }else if(num[lowIndex] + num[currentIndex] + num[highIndex] < 0){
                    lowIndex++;
                }else{
                    highIndex--;
                }
            }
        }
        return result;
    }
    
    /*
     * leetcode valid parentheses
     */
    public boolean isValid(String s) {
        Stack<Character> myStack = new Stack<Character>();
        for(int i=0; i<s.length();++i){
            if(s.charAt(i) == '(' || s.charAt(i) == '[' || s.charAt(i) == '{'){
                myStack.push(s.charAt(i));
            }else{
                if(myStack.empty() || myStack.peek() != revertChar(s.charAt(i))){
                    return false;
                }
                else if(myStack.peek() == revertChar(s.charAt(i))){
                    myStack.pop();
                }
            }
        }
        return myStack.empty();
    }
    
    private char revertChar(char c){
        switch(c){
            case '}': return '{';
            case ')': return '(';
            case ']': return '[';
        }
        return ' ';
    }
    
    /*
     * leetcode merge 2 sorted list
     */ 

    public ListNode mergeTwoLists(ListNode l1, ListNode l2) {
        if(l1 ==null && l2 == null){
            return null;
        }
        ListNode resultHead = new ListNode(0);
        ListNode travel = resultHead;
        while(l1 != null && l2 !=null){
            if(l1.val > l2.val){
                travel.next = new ListNode(l2.val);
                l2 = l2.next;
            }else{
                travel.next = new ListNode(l1.val);
                l1 = l1.next;
            }
            travel = travel.next;
        }
        while(l1 != null){
            travel.next = new ListNode(l1.val);
            l1 = l1.next;
            travel = travel.next;
        }
        while(l2 != null){
            travel.next = new ListNode(l2.val);
            l2 = l2.next;
            travel = travel.next;
        }
        return resultHead.next;
    }
    
    public void assignValue(ListNode travel, int val, ListNode head){
    	if(travel == null){
    		travel = new ListNode(val);
    	}else{
    		travel.next = new ListNode(val);
    		travel = travel.next;
    	}
    }
    private class ListNode {
        int val;
        ListNode next;
        ListNode(int x) {
            val = x;
            next = null;
        }
    }
    
    public String strStr(String haystack, String needle) {
        // Note: The Solution object is instantiated only once and is reused by each test case.
        int hayStart = 0;
        int hayLen = haystack.length();
        int needleLen = needle.length();
        while(hayStart <= hayLen-needleLen){
            int hayTravel = hayStart;
            int needleTravel = 0;
            while(needleTravel < needleLen && haystack.charAt(hayTravel) == needle.charAt(needleTravel)){
                hayTravel++;
                needleTravel++;
            }
            if(needleTravel == needleLen){
                return haystack.substring(hayStart);
            }
            hayStart++;
        }
        return null;
    }
    
    /*
     * leetcode n queens
     * 
     */
    
    ArrayList<String[]> sol = new ArrayList<String[]>();
    public ArrayList<String[]> solveNQueens(int n) {
        // Start typing your Java solution below
        // DO NOT write main() function
        ArrayList<String[]> result = new ArrayList<String[]>();
        int[] rowPosArray = new int[n];
        if(n == 2 || n == 3){
            return result;
        }
        solve(0,n,rowPosArray);
        return sol;
    }
    
    public String[] print(int[] seq, int n){
        String[] res = new String[n];
        for(int i=0;i<n;++i){
            StringBuilder sb = new StringBuilder();
            for(int j=0;j<n;++j){
                if(seq[i] == j){
                    sb.append('Q');
                }else{
                    sb.append('.');
                }
            }
            res[i] = sb.toString();
        }
        return res;
    }
    
    public boolean isValid(int currentQueen, int rowPos, int[] rowPosArray){
        for(int i=0;i<currentQueen; ++i){
            if(rowPos == rowPosArray[i] ||
                rowPos - rowPosArray[i] == currentQueen - i ||
                rowPos - rowPosArray[i] == i - currentQueen){
                    return false;
                }
        }
        return true;
    }
    
    public void solve(int startQueen, int n, int[] rowPosArray){
        if(startQueen == n){
            //problem solved
            sol.add(print(rowPosArray,n));
        }else{
            for(int i=0;i<n;++i){
                if(isValid(startQueen,i,rowPosArray)){
                    rowPosArray[startQueen] = i;
                    solve(startQueen+1,n, rowPosArray);
                }
            }
        }
    }
    
    /*
     * leetcode pow(x,n)
     */
    public double pow(double x, int n) {
        // Start typing your Java solution below
        // DO NOT write main() function
        if(n == 0){
            return 1.0;
        }
        double half = pow(x,n/2);
        if(n%2 == 0){
            return half * half;
        }else if(n>0){
            return half * half*x;
        }else{
            return half * half/x;
        }
    }
    
    /*
     * careers cup maximum subarray problem:
     * the maximum subarray problem is the task of 
     * finding the contiguous subarray within a one-dimensional array of numbers (containing at least one positive number) which has the largest sum. 
     * 
     */
    public int maxSubArray(int[] input){
    	if(input.length == 0){
    		return 0;
    	}
    	if(input.length == 1){
    		return input[0] > 0 ? input[0] : 0;
    	}
    	int maxSoFar = 0;
    	int maxEndingHere = 0;
    	for(int i=1;i<input.length;++i){
    		//int[] currentSubArray = new int[i];
    		//System.arraycopy(input, 0, currentSubArray, 0, i);
    		maxEndingHere = Math.max(0, maxEndingHere+input[i]);
    		maxSoFar = Math.max(maxSoFar, maxEndingHere);
    	}
    	return maxSoFar;
    }
    
    /*
     * coupons.com onsite questions:
     * given a set of integers, then give an input n, find ways of composition of n that only use numbers inside the set
     * for example, give a set of integers : int[] numberSet = {3,7,11,13,17} and n = 16
     * the output should be {3,3,3,7} and {3,13}
     * this implementation has bug!!
     */ 
    void printSelectionWays(int[] numberSet, int n, ArrayList<Integer> prefix, ArrayList<ArrayList<Integer>> result){
    	int setLengh = numberSet.length;
    	for(int i=0;i<setLengh;++i){
    		if(n%numberSet[i] == 0){
    			//find solution
    			while(n>0){
    				prefix.add(numberSet[i]);
    				n -= numberSet[i];
    			}
    			for(int j=0;j<prefix.size();j++){
    				System.out.print(prefix.get(j));
    			}

    		}
    	}
    	if(n>0){
    		for(int i=0;i<setLengh;++i){
    			if(n > numberSet[i]){
    				n -= numberSet[i];
    				prefix.add(numberSet[i]);
    				printSelectionWays(numberSet,n,prefix,result);
    			}
    		}
    	}
    	
    } 
    
    public int numberOfWays(int[] coinSet, int useLength, int remainValue){
    	if(useLength<1 || remainValue < 0){
    		return 0;
    	}
    	if(remainValue == 0){
    		return 1;
    	}
    	else{
    		return numberOfWays(coinSet,useLength-1, remainValue) + numberOfWays(coinSet,useLength,remainValue-coinSet[useLength-1]);
    	}
    }
    
    /*
     * leetcode anagrams
     * 
     */
    public ArrayList<String> anagrams(String[] strs) {
        // Start typing your Java solution below
        // DO NOT write main() function
        HashMap<String,ArrayList<String>> anagramMap = new HashMap<String,ArrayList<String>>();
        for(int i=0;i<strs.length;++i){
            String sortedString = getSortedString(strs[i]);
            ArrayList<String> currentList = anagramMap.get(sortedString);
            if(currentList != null){
                currentList.add(strs[i]);
            }else{
                currentList = new ArrayList<String>();
                currentList.add(strs[i]);
                anagramMap.put(sortedString, currentList);
            }
        }
        ArrayList<String> res = new ArrayList<String>();
	    for (ArrayList<String> strList : anagramMap.values()) {
	        if (strList.size() > 1) {
	            res.addAll(strList);
	        }           
        }
        return res;
    }
    
    public String getSortedString(String s){
        char[] charArray = s.toCharArray();
        Arrays.sort(charArray);
        return(new String(charArray));
        
    }
    
    /*
     * leetcode insert Interval
     */
    
    public class Interval {
    	      int start;
    	      int end;
    	      Interval() { start = 0; end = 0; }
    	      Interval(int s, int e) { start = s; end = e; }
    	  }
    public ArrayList<Interval> insert(ArrayList<Interval> intervals, Interval newInterval) {
        // Start typing your Java solution below
        // DO NOT write main() function
        ArrayList<Interval> result = new ArrayList<Interval>();
        if(intervals.size() == 0){
            result.add(newInterval);
            return result;
        }
        Collections.sort(intervals,comparator);
        boolean newIntervalAdded = false;
        for(int i=0;i<intervals.size(); ++i){
            Interval currentInterval = intervals.get(i);
            
            if(currentInterval.end < newInterval.start){
                result.add(currentInterval);
            }else if(newInterval.end < currentInterval.start){
                if(!newIntervalAdded){
                    result.add(newInterval);
                }
                result.add(currentInterval);
                newIntervalAdded = true;
            }else{
                newInterval.start = Math.min(newInterval.start,currentInterval.start);
                newInterval.end = Math.max(newInterval.end,currentInterval.end);
            }
             
        }
        if(!newIntervalAdded){
            result.add(newInterval);
        }
        return result;
    }
    
    public Comparator<Interval> comparator = new Comparator<Interval>(){
        public int compare(Interval l1, Interval l2){
            if(l1.start < l2.start){
                return -1;
            }else if(l1.start > l2.start){
                return 1;
            }else{
                if(l1.end < l2.end){
                    return -1;
                }else if(l1.end > l2.end){
                    return 1;
                }else{
                    return 0;
                }
            }
        }
    };
    
    
    public ArrayList<Interval> merge(ArrayList<Interval> intervals) {
        // Start typing your Java solution below
        // DO NOT write main() function
        
        Collections.sort(intervals,comparator);
        ArrayList<Interval> result = new ArrayList<Interval>();
        if(intervals.size() == 0){
            return result;
        }
        result.add(intervals.get(0));
        for(int i=1;i<intervals.size();++i){
            Interval lastAdded = result.get(result.size() - 1);
            Interval intervalToBeAdded = intervals.get(i);
            if(lastAdded.end < intervalToBeAdded.start){
                result.add(intervalToBeAdded);
            }else{
            	result.remove(lastAdded);
                intervalToBeAdded.start = lastAdded.start;
                intervalToBeAdded.end = Math.max(lastAdded.end,intervalToBeAdded.end);
                result.add(intervalToBeAdded);
            }
        }
        return result;
    }
    
    /*
     * leetcode generate parenthesise recursive, has bug when n = 4, failed case: (())(())
     * 
     */
    public ArrayList<String> generateParenthesisRecursive(int n) {
        // Start typing your Java solution below
        // DO NOT write main() function
        ArrayList<String> result = new ArrayList<String>();
        if(n == 0){
            return result;
        }
        if(n == 1){
            result.add("()");
            return result;
        }
        ArrayList<String> previous = generateParenthesis(n-1);
        Map<String, Integer> checkDup = new HashMap<String,Integer>();
        for(int i=0;i<previous.size();++i){
            StringBuilder sb1 = new StringBuilder();
            sb1.append("()");
            sb1.append(previous.get(i));
            String cadidate1 = sb1.toString();
            if(checkDup.get(cadidate1) == null){
                result.add(cadidate1);
                checkDup.put(cadidate1,1);
            }
            
            StringBuilder sb2 = new StringBuilder();
            sb2.append(previous.get(i));
            sb2.append("()");
            String cadidate2 = sb2.toString();
            if(checkDup.get(cadidate2) == null){
                result.add(cadidate2);
                checkDup.put(cadidate2,1);
            }
            
            StringBuilder sb3 = new StringBuilder();
            sb3.append("(");
            sb3.append(previous.get(i));
            sb3.append(")");
            String cadidate3 = sb3.toString();
            if(checkDup.get(cadidate3) == null){
                result.add(cadidate3);
                checkDup.put(cadidate3,1);
            }
        }
        return result;
    }
    
    public ArrayList<String> generateParenthesis(int n) {
        // Start typing your Java solution below
        // DO NOT write main() function
        ArrayList<String> result = new ArrayList<String>();
        generator(result,"",0,0,n);
        return result;
    }
    
    public void generator(ArrayList<String> result, String curr, int left, int right, int n){
        if(left == n){
            //find a solution
            while(right < n){
                curr = curr + ")";
                right++;
            }
            result.add(curr);
            return;
        }
        generator(result,curr+"(",left+1,right,n);
        if(left>right){
            generator(result,curr+")",left,right+1,n);
        }
    }
    
    public boolean isValidBST(TreeNode root) {
        // Start typing your Java solution below
        // DO NOT write main() function
        return isValidBST(root, Integer.MIN_VALUE, Integer.MAX_VALUE);
    }
    
    private boolean isValidBST(TreeNode root, int low, int high){
        if(root == null){
            return true;
        }
        if(!isValidBST(root.left,low, root.val) || !isValidBST(root.right,root.val,high)){
            return false;
        }
        if(root.val > high || root.val == high || root.val < low || root.val == low){
            return false;
        }
        return true;
    }
    
    public class TreeNode {
      int val;
      TreeNode left;
      TreeNode right;
      TreeNode(int x) { val = x; }
    }
    
    /*
     * leetcode canJump, recursive solution, not efficient.
     */
    /*
    public boolean canJump(int[] A) {
        // Start typing your Java solution below
        // DO NOT write main() function
        return canJumpHelper(A,A.length-1);
    }
    
    public boolean canJumpHelper(int[] A, int targetIndex){
        if(A[0] >= targetIndex){
            return true;
        }
        for(int i=1;i<=targetIndex-1;++i){
            if(A[i] >= targetIndex-i && canJumpHelper(A,i)){
                return true;
            }
        }
        return false;
    }
    */
    /*
     * jump game dp solution
     */
    public boolean canJump(int[] A) {
        // Start typing your Java solution below
        // DO NOT write main() function
        Map<Integer,Boolean> canJumpTrack = new HashMap<Integer,Boolean>();
        return canJumpHelper(A,canJumpTrack);
    }
    
    public boolean canJumpHelper(int[] A, Map<Integer,Boolean> myMap){
        myMap.put(0,true);
        int targetIndex = A.length-1;
        for(int i=1;i<=targetIndex;++i){
            if(A[0] > i-1){
                myMap.put(i,true);
            }else{
                for(int j=0;j<i;++j){
                    if(A[j] > i-1-j && myMap.get(j)){
                        myMap.put(i,true);
                    }
                }
                if(myMap.get(i) == null){
                	return false;
                }
            }
        }
        return (myMap.get(targetIndex) == null)  ? false : true;
    }
    
    /*
     * leetcode wildcard match recursive
     */
    public boolean isMatch(String s, String p) {
        // Start typing your Java solution below
        // DO NOT write main() function
        return isMatch(s,p,0,0);
    }
    public boolean isMatch(String s, String p, int sStart, int pStart){
        if(pStart == p.length() || sStart == s.length()){
            return pStart == p.length() && sStart == s.length();
        }
        if(p.charAt(pStart) == '*'){
            while(pStart != p.length() && p.charAt(pStart) == '*'){
                pStart++;
            }
            if(pStart == p.length()){
                return true;
            }
            while(sStart != s.length() && !isMatch(s,p,sStart,pStart)){
                sStart++;
            }
            return sStart != s.length();
        }
        if(p.charAt(pStart) == '?' || p.charAt(pStart) == s.charAt(sStart)){
            return isMatch(s,p,sStart+1,pStart+1);
        }
        return false;
    }
    /*
     * leetcode delete duplicate number in linked list
     * 
     */
    public ListNode deleteDuplicates(ListNode head) {
        // Start typing your Java solution below
        // DO NOT write main() function
        if(head == null){
            return null;
        }
        Map<Integer,Boolean> numberMap = new HashMap<Integer,Boolean>();
        ListNode travel = head;
        numberMap.put(travel.val, true);
        while(travel != null && travel.next != null){
            if(numberMap.get(travel.next.val) == null){
                numberMap.put(travel.next.val, true);
                travel = travel.next;
            }
            else{
                travel.next = travel.next.next;
                travel = travel.next;
            }
        }
        return head;
    }
    
    public int ladderLength(String start, String end, HashSet<String> dict) {
        // Start typing your Java solution below
        // DO NOT write main() function
        if(start.equals(end)){
            return 1;
        }
        
        int startLength = 2;
        Map<String, Boolean> visitMap = new HashMap<String, Boolean>();
        Queue<String> currentLevelQueue = new LinkedList<String>();
        Queue<String> nextLevelQueue = new LinkedList<String>();
        currentLevelQueue.add(start);

        while(currentLevelQueue.size() != 0){
            String curr = currentLevelQueue.remove();
            for(int i=0;i<start.length();++i){
                for(int j=0;j<26;++j){
                    StringBuilder sb = new StringBuilder(curr);
                    sb.setCharAt(i,(char)('a'+j));
                    String morphString = sb.toString();
                    if(end.equals(morphString)){
                        return startLength;
                    }
                    if(dict.contains(morphString) && visitMap.get(morphString) == null){
                        visitMap.put(morphString,true);
                        nextLevelQueue.add(morphString);
                    }
                    
                }
            }
            if(currentLevelQueue.size() == 0 && nextLevelQueue.size() != 0){
                currentLevelQueue = nextLevelQueue;
                nextLevelQueue = new LinkedList<String>();
                startLength++;
            }
        }
        return 0;
    }
    
    public int maxProfit(int[] prices) {
        // Start typing your Java solution below
        // DO NOT write main() function
        int max = 0;
        if(prices.length == 0){
            return max;
        }
        int minimum = prices[0];
        for(int i = 1;i < prices.length;++i){
            int currentProfit = prices[i] - minimum;
            if(currentProfit > max){
                max = currentProfit;
            }
            if(prices[i] < minimum){
                minimum = prices[i];
            }
        }
        return max;
    }
    
    public boolean NumberisPalindrome(int x) {
        // Start typing your Java solution below
        // DO NOT write main() function
    	if(x<0){
    		return false;
    	}
        int digits = 1;
        while(x/digits >= 10){
        	digits *= 10;
        }
        while(x > 0){
        	int high = x/digits;
        	int low = x%10;
        	if(high != low){
        		return false;
        	}
        	x = x - high*digits;
        	x = x/10;
        	digits = digits/100;
        }
        return true;
    }
    
    public String longestPalindrome(String s) {
        // Start typing your Java solution below
        // DO NOT write main() function
        if(s.length() == 0){
            return "";
        }
        String longest = s.substring(0,1);
        for(int i=1;i<s.length();++i){
            String longest1 = longestPalindromeAroundCenter(s,i,i);
            if(longest1.length() > longest.length()){
                longest = longest1;
            }
            longest1 = longestPalindromeAroundCenter(s,i-1,i);
            if(longest1.length() > longest.length()){
                longest = longest1;
            }
        }
        return longest;
    }
    
    public String longestPalindromeAroundCenter(String s, int left, int right){
        //check center at n
        int sLen = s.length();
        while(left>=0 && right<sLen && s.charAt(left) == s.charAt(right)){
            left--;
            right++;
        }
        return s.substring(left+1,right);
        
        
    }
    
    public int lengthOfLongestSubstring(String s) {
        // Start typing your Java solution below
        // DO NOT write main() function
        int sLen = s.length();
        if(sLen == 0 || sLen == 1){
            return sLen;
        }
        int maxLen = 1;
        int startIndex = 0;
        int endIndex = 1;
        Map<Character,Boolean> repeatMap = new HashMap<Character,Boolean>();
        repeatMap.put(s.charAt(startIndex),true);
        while(endIndex != sLen){
            if(repeatMap.get(s.charAt(endIndex)) == null){
                repeatMap.put(s.charAt(endIndex),true);
                endIndex++;
            }else{
                int currLen = endIndex-startIndex;
                if(currLen > maxLen){
                    maxLen = currLen;
                }
                repeatMap.clear();
                startIndex = endIndex;
                repeatMap.put(s.charAt(startIndex),true);
                endIndex = endIndex + 1;
            }
        }
        int currLen = endIndex-startIndex;
        if(currLen > maxLen){
            maxLen = currLen;
        }
        return maxLen;
    }
    
    public ArrayList<ArrayList<Integer>> permute(int[] num) {
        // Start typing your Java solution below
        // DO NOT write main() function
        ArrayList<ArrayList<Integer>> result = new ArrayList<ArrayList<Integer>>();
        ArrayList<Integer> remaining = new ArrayList<Integer>();
        ArrayList<Integer> preList = new ArrayList<Integer>();
        for(int i=0;i<num.length;++i){
            remaining.add(num[i]);
        }
        permuteHelper(remaining,preList,result);
        return result;
    }
    
    public void permuteHelper(ArrayList<Integer> remainingNum, ArrayList<Integer> preList, ArrayList<ArrayList<Integer>> result)              {
        if(remainingNum.size() == 0){
            result.add(preList);
            return;
        }
        for(int i=0;i<remainingNum.size(); ++i){
            ArrayList<Integer> newPreList = new ArrayList<Integer>(preList);
            ArrayList<Integer> newRemaining = new ArrayList<Integer>(remainingNum);
            newPreList.add(remainingNum.get(i));
            newRemaining.remove(i);
            permuteHelper(newRemaining,newPreList,result);
        }
    }
    
    public ArrayList<ArrayList<String>> partition(String s) {
        // Start typing your Java solution below
        // DO NOT write main() function
        ArrayList<ArrayList<String>> result = new ArrayList<ArrayList<String>>();
        if(s.length() == 0){
            return result;
        }
        ArrayList<String> preList = new ArrayList<String>();
        partitionHelper(s,preList,0,result);
        return result;
    }
    
    public boolean isPalindrome(String s){
        if(s.length() == 0){
            return false;
        }
        int start = 0;
        int end = s.length() -1;
        while(start < end){
            if(s.charAt(start) != s.charAt(end)){
                return false;
            }
            start++;
            end--;
        }
        return true;
    }
    
    public void partitionHelper(String s, ArrayList<String> preList, int currentIndex, ArrayList<ArrayList<String>> result){
        if(currentIndex == s.length()){
            result.add(preList);
        }
        for(int nextIndex = currentIndex+1;nextIndex <= s.length(); ++ nextIndex){
            if(isPalindrome(s.substring(currentIndex,nextIndex))){
                ArrayList<String> newPreList = new ArrayList<String>(preList);
                newPreList.add(s.substring(currentIndex,nextIndex));
                partitionHelper(s,newPreList,nextIndex,result);
            }
        }
    }
    /*
     * leetcode sum root to leaf numbers
     */
    public int sumNumbers(TreeNode root) {
        // Start typing your Java solution below
        // DO NOT write main() function
        if(root == null){
            return 0;
        }
        int result = sumNumbersHelper(root, 0);
        return result;
    }
    
    public int sumNumbersHelper(TreeNode root, int parentSum){
        parentSum = parentSum *10 + root.val;
        int left = 0;
        int right = 0;
        if(root.left != null){
            left = sumNumbersHelper(root.left,parentSum);
        }
        if(root.right != null){
            right = sumNumbersHelper(root.right,parentSum);
        }
        if(left + right > 0){
            return left + right;
        }else{
            return parentSum;
        }
    }
    
    public int maxArea(int[] height) {
        // Start typing your Java solution below
        // DO NOT write main() function
        int startIndex = 0;
        int endIndex = height.length -1;
        int currentMax = 0;
        while(startIndex < endIndex){
            int currentArea = 0;
            if(height[startIndex] <= height[endIndex]){
                currentArea = (endIndex-startIndex) * height[startIndex];
                startIndex++;
            }else{
                currentArea = (endIndex-startIndex) * height[endIndex];
                endIndex--;
            }
            if(currentArea > currentMax){
                    currentMax = currentArea;
            }
        }
        return currentMax;
    }
    
    
    public int trap(int[] A) {
        // Start typing your Java solution below
        // DO NOT write main() function
        int alen = A.length;
        if(alen <=2){
            return 0;
        }
        int trapSum = 0;
        int left = 0;
        int right = alen -1;
        int runner = 0;
        while(left<right){
            if(A[left] >= A[right]){
                runner = right-1;
                while(A[right] > A[runner]){
                    trapSum += A[right] - A[runner];
                    runner--;
                }
                right = runner;
            }else{
               runner = left + 1;
               while(A[left] > A[runner]){
                   trapSum += A[left] - A[runner];
                   runner++;
               }
               left = runner;
            }
            
        }
        return trapSum;
    }
    
    public int testCharCal(char c){
    	return c-'0';
    }
    
    public String addBinary(String a, String b) {
        // Start typing your Java solution below
        // DO NOT write main() function
        return a.length() > b.length() ? addBinaryHelper(a,b) : addBinaryHelper(b,a);
    }
    
    public String addBinaryHelper(String longString, String shortString){
        int carry = 0;
        int longIndex = longString.length()-1;
        int shortIndex = shortString.length()-1;
        StringBuilder sb = new StringBuilder();
        while(shortIndex >=0){
            int sum = carry + (longString.charAt(longIndex)-'0') + (shortString.charAt(shortIndex) - '0');
            carry = sum / 2;
            sum = sum %2;
            sb.insert(0,sum);
            shortIndex--;
            longIndex--;
        }
        while(longIndex >=0){
            int sum = carry + longString.charAt(longIndex)-'0';
            carry = sum/2;
            sum = sum%2;
            sb.insert(0,sum);
            longIndex--;
        }
        if(carry > 0){
            sb.insert(0,'1');
        }
        return sb.toString();
    }
    
    public ArrayList<ArrayList<Integer>> combine(int n, int k) {
        // Start typing your Java solution below
        // DO NOT write main() function
        ArrayList<ArrayList<Integer>> result = new ArrayList<ArrayList<Integer>>();
        //assume k<=n here
        if(k == 0 || k>n){
            return result;
        }
        if(k == 1){
            for(int i=1;i<=n;++i){
                ArrayList<Integer> currList = new ArrayList<Integer>();
                currList.add(i);
                result.add(currList);
            }
            return result;
        }
        ArrayList<ArrayList<Integer>> result1 = combine(n-1,k);
        ArrayList<ArrayList<Integer>> result2 = combine(n-1,k-1);
        for(int i=0; i<result1.size();++i){
            ArrayList<Integer> currList = result1.get(i);
            result.add(currList);
        }
        for(int i=0; i<result2.size();++i){
            ArrayList<Integer> currList = result2.get(i);
            currList.add(n);
            result.add(currList);
        }
        return result;
    }
    
    public int minPathSum(int[][] grid) {
        // Start typing your Java solution below
        // DO NOT write main() function
        int rowLen = grid.length;
        int colLen = grid[0].length;
        int[][] pathSumMap = new int[rowLen+1][colLen+1];
        for(int i=0;i<rowLen+1;++i){
            pathSumMap[i][0] = Integer.MAX_VALUE;
        }
        for(int i=1;i<colLen+1;++i){
            pathSumMap[0][i] = Integer.MAX_VALUE;
        }

        for(int i=1;i<rowLen+1;++i){
            for(int j=1;j<colLen+1;++j){
                if(i == 1 && j==1){
                    pathSumMap[i][j] = grid[0][0];
                }else{
                    pathSumMap[i][j] = pathSumMap[i-1][j] > pathSumMap[i][j-1] ? pathSumMap[i][j-1]+grid[i-1][j-1] : pathSumMap[i-1][j] +grid[i-1][j-1];
                }
            }
        }
        return pathSumMap[rowLen][colLen];
    }
    
    public int candy(int[] ratings) {
        // Note: The Solution object is instantiated only once and is reused by each test case.
        int ratingLen = ratings.length;
        if(ratingLen <= 1){
            return ratingLen;
        }
        int[] assignArray = new int[ratingLen];
        assignArray[0] = 1;
        for(int i=1;i<ratingLen;++i){
            if(ratings[i] > ratings[i-1]){
                assignArray[i] = assignArray[i-1] +1;
            }else{
                assignArray[i] =1;
                int j = i;
                while(ratings[j-1] > ratings[j] && assignArray[j-1] <= assignArray[j] && j>0){
                    assignArray[j-1] = assignArray[j] +1;
                    j--;
                }
            }
        }
        int minValue = 0;
        for(int i=0;i<ratingLen; ++i){
            minValue += assignArray[i];
        }
        return minValue;
    }
    public ListNode partition(ListNode head, int x) {
        // Start typing your Java solution below
        // DO NOT write main() function
        ListNode result;
        ListNode small = new ListNode(0);
        ListNode big = new ListNode(0);
        ListNode travelsmall = small;
        ListNode travelbig = big;
        while(head != null){
            if(head.val >= x){
                travelbig.next = head;
                travelbig = travelbig.next;
            }else{
                travelsmall.next = head;
                travelsmall = travelsmall.next;
            }
            head = head.next;
        }
        travelsmall.next = big.next;
        result = small.next;
        return result;
    }
    
    int maxSum = Integer.MIN_VALUE;
    public int maxPathSum(TreeNode root) {
        // Note: The Solution object is instantiated only once and is reused by each test case.
        int csum = 0;
        maxPathSumHelper(root,csum);
        return maxSum;
    }
    
    public int maxPathSumHelper(TreeNode root, int csum){
        if(root == null){
            return 0;
        }
        int leftSum = maxPathSumHelper(root.left,0);
        int rightSum = maxPathSumHelper(root.right,0);
        csum = Math.max(Math.max(leftSum + root.val,rightSum + root.val),root.val);
        maxSum = Math.max(maxSum,Math.max(csum, root.val + leftSum+rightSum));
        return csum;
    }
    
    class UndirectedGraphNode {
    	      int label;
    	      ArrayList<UndirectedGraphNode> neighbors;
    	      UndirectedGraphNode(int x) { label = x; neighbors = new ArrayList<UndirectedGraphNode>(); }
    	  };
    public UndirectedGraphNode cloneGraph(UndirectedGraphNode node) {
        // Note: The Solution object is instantiated only once and is reused by each test case.
        Map<UndirectedGraphNode,UndirectedGraphNode> nodeMap = new HashMap<UndirectedGraphNode,UndirectedGraphNode>();
        Set<UndirectedGraphNode> visitedNode = new HashSet<UndirectedGraphNode>();
        List<UndirectedGraphNode> visitList = new ArrayList<UndirectedGraphNode>();
        if(node == null){
            return null;
        }
        visitList.add(node);
        while(visitList.size() != 0){
            UndirectedGraphNode currNode = visitList.remove(0);
            if(visitedNode.contains(currNode)){
                continue;
            }
            UndirectedGraphNode clonedNode = nodeMap.get(currNode);
            if(clonedNode == null){
                clonedNode = new UndirectedGraphNode(node.label);
                nodeMap.put(currNode,clonedNode);
            }
           
            for(int i=0;i<currNode.neighbors.size(); ++i){
                UndirectedGraphNode currNeighbor = currNode.neighbors.get(i);
                UndirectedGraphNode clonedNeighbor = nodeMap.get(currNeighbor);
                if(clonedNeighbor == null){
                    clonedNeighbor = new UndirectedGraphNode(currNeighbor.label);
                    nodeMap.put(currNeighbor,clonedNeighbor);
                }
                clonedNode.neighbors.add(clonedNeighbor);
                visitList.add(currNeighbor);
            }
            visitedNode.add(currNode);
        }
        return nodeMap.get(node);
    }
    class RandomListNode {
    	      int label;
    	      RandomListNode next, random;
    	      RandomListNode(int x) { this.label = x; }
    	  }
    
    public RandomListNode copyRandomList(RandomListNode head) {
        // Note: The Solution object is instantiated only once and is reused by each test case.
        Map<RandomListNode, RandomListNode> nodeMap = new HashMap<RandomListNode,RandomListNode>();
        RandomListNode headOrigin = head;
        while(head != null){
            RandomListNode cloneNode = new RandomListNode(head.label);
            nodeMap.put(head,cloneNode);
            head = head.next;
        }
        head = headOrigin;
        while(head != null){
            RandomListNode cloneNode = nodeMap.get(head);
            RandomListNode nextNode = head.next;
            cloneNode.next = nodeMap.get(nextNode);
            RandomListNode ranNode = head.random;
            cloneNode.random = nodeMap.get(ranNode);
            head = head.next;
        }
        return nodeMap.get(headOrigin);
    }
    
    
    public void solveSudoku(char[][] board) {
        // Start typing your Java solution below
        // DO NOT write main() function
        board = solveHelper(board,0,0);
        
    }
    
    public boolean checkValid(char[][] board, int row, int col, char candidate){
        //check row
        for(int i=0;i<9;++i){
            if(i == col){
                continue;
            }
            if(board[row][i] == candidate){
                return false;
            }
        }
        //check col
        for(int i=0;i<9;++i){
            if(i == row){
                continue;
            }
            if(board[i][col] == candidate){
                return false;
            }
        }
        //check square
        int startRow = row/3;
        int startCol = col/3;
        for(int i=startRow;i<startRow+3;++i){
            for(int j=startCol; j<startCol+3;++j){
                if(i == row && j == col){
                    continue;
                }
                if(board[i][j] == candidate){
                    return false;
                }
            }
        }
        return true;
    }
    public char[][] solveHelper(char[][] board, int currRow, int currCol){
        if(board[currRow][currCol] != '.'){
            solveHelper(board,currRow,currCol+1);
        }
        if(currCol>=9){
            currRow += 1;
            currCol = 0;
        }
        if(currRow>=9){
            return board;
        }
        for(int i=1;i<=9;++i){
            if(checkValid(board, currRow,currCol,(char)(i+'0'))){
                board[currRow][currCol] = (char)(i+'0');
                board = solveHelper(board,currRow,currCol+1);
            }
        }
        return board;
    }
    
    public ArrayList<String> letterCombinations(String digits) {
        // Start typing your Java solution below
        // DO NOT write main() function
        String[] numberMap = {"abc","def","ghi","jkl","mno","pqrs","tuv","wxyz"};
        ArrayList<String> result = new ArrayList<String>();
        letterComboHelper(digits,"",0,numberMap,result);
        return result;
    }
    
    public void letterComboHelper(String digits, String pre, int currIndex, String[] dic, ArrayList<String> res){
        if(currIndex == digits.length()){
            res.add(pre);
            return;
        }
        int currNum = digits.charAt(currIndex) - '0';
        for(int i=0;i<dic[currNum-2].length();++i){
            pre = pre + dic[currNum-2].charAt(i);
            letterComboHelper(digits,pre,currIndex+1,dic,res);
        }
    }
    
    public TreeNode sortedListToBST(ListNode head) {
        // Start typing your Java solution below
        // DO NOT write main() function
        if(head == null){
            return null;
        }
        TreeNode root = listToBSTHelper(head, null);
        return root;
    }
    
    public TreeNode listToBSTHelper(ListNode head, ListNode end){
        if(head == null){
            return null;
        }
        if(head != null && head.next == end){
            return new TreeNode(head.val);
        }

        ListNode slow = head;
        ListNode faster = head;
        while(faster != null && faster.next != null && faster!= end && faster.next != end){
            slow = slow.next;
            faster = faster.next.next;
        }
        TreeNode root = new TreeNode(slow.val);
        root.left = listToBSTHelper(head,slow);
        root.right = listToBSTHelper(slow.next,null);
        return root;
    }
    
    public ListNode mergeKLists(ArrayList<ListNode> lists) {
        // Start typing your Java solution below
        // DO NOT write main() function
        int kLen = lists.size();
        if(kLen == 0){
            return null;
        }
        while(kLen > 1){
            ArrayList<ListNode> nextList = new ArrayList<ListNode>();
            int i=0;
            for(i=0;i<kLen-1;i = i+2){
                ListNode nextListNode = merge2Lists(lists.get(i),lists.get(i+1));
                nextList.add(nextListNode);
            }
            if(i == kLen-1){
                nextList.add(lists.get(i));
            }
            lists = nextList;
            kLen = lists.size();
        }
        return lists.get(0);
    }
    public ListNode merge2Lists(ListNode list1, ListNode list2){
        ListNode result = new ListNode(0);
        while(list1 != null && list2 != null){
            if(list1.val < list2.val){
                result.next = new ListNode(list1.val);
                list1 = list1.next;
            }else{
                result.next = new ListNode(list2.val);
                list2 = list2.next;
            }
            result = result.next;
        }
        if(list1 != null){
            result.next = list1;
        }
        if(list2 != null){
            result.next = list2;
        }
        return result.next;
    }
    
    
    public String getPermutation(int n, int k) {
        // Start typing your Java solution below
        // DO NOT write main() function
        if(n == 1){
            return "1";
        }
        
        Set<Integer> selectedNumberSet = new HashSet<Integer>();
        StringBuffer sb = new StringBuffer();
        ArrayList<Integer> permutationCount = getPermutationCount(n);
        return getPermutationHelper(n,k,selectedNumberSet,sb,permutationCount);
    }
    
    public String getPermutationHelper(int n, int k, Set<Integer> numberSet, StringBuffer preBuffer, ArrayList<Integer> permutationDic){
        int preLen = preBuffer.length();
        if(preLen == n){
            return  preBuffer.toString();
        }
        if(k == 1){
            for(int i=0;i<n-preLen;++i){
                int firstElement = findkthNumber(n,numberSet,1);
                preBuffer.append(Character.toChars(firstElement + '0'));
                numberSet.add(firstElement);
            }
            return preBuffer.toString();
        }
        int restPermutationLen = n - preLen;
        for(int i=1;i<=restPermutationLen;++i){
            if(k <= i * permutationDic.get(restPermutationLen-2)){
                k = k - (i-1)*permutationDic.get(restPermutationLen-2);
                int numberToAdd = findkthNumber(n,numberSet,i);
                numberSet.add(numberToAdd);
                preBuffer.append(Character.toChars(numberToAdd + '0'));
                return getPermutationHelper(n,k,numberSet,preBuffer,permutationDic);
            }
        }
        //should not reach here
        return null;
    }
    
    
    public int findkthNumber(int n, Set<Integer> numberSet, int k){
        for(int i=1;i<=n;++i){
            if(numberSet.contains(i)){
                continue;
            }
            k--;
            if(k == 0){
                return i;
            }
        }
        //should not fall into this case
        return 0;
    }
    
    public ArrayList<Integer> getPermutationCount(int n){
        ArrayList<Integer> result = new ArrayList<Integer>();
        int curr = 1;
        for(int i=1; i<=n;++i){
            curr = curr *i;
            result.add(curr);
        }
        return result;
    }
    
    public boolean isNumber(String s) {
        // Start typing your Java solution below
        // DO NOT write main() function
        int[][] transitionMap = {
            { 0, 1, 3,-1, 2,-1},
            { 4, 1, 3, 5,-1,-1},
            {-1, 1, 3,-1,-1,-1},
            {-1, 6,-1,-1,-1,-1},
            { 4,-1,-1,-1,-1,-1},
            {-1, 7,-1,-1, 8,-1},
            { 4, 6,-1, 5,-1,-1},
            { 4, 7,-1,-1,-1,-1},
            {-1, 7,-1,-1,-1,-1}
        };
        int sLen = s.length();
        int state = 0;
        int travel = 0;
        int transitionType = -1;
        while(travel != sLen){
            char currChar = s.charAt(travel);
            if(currChar == ' '){
                transitionType = 0;
            }
            else if(currChar>= '0' && currChar <= '9'){
                transitionType = 1;
            }
            else if (currChar == '.'){
                transitionType = 2;
            }
            else if(currChar == 'e' || currChar == 'E'){
                transitionType = 3;
            }
            else if(currChar == '-' || currChar == '+'){
                transitionType = 4;
            }
            else{
                transitionType = 5;
            }
            state = transitionMap[state][transitionType];
            if(state == -1){
                return false;
            }
            travel++;
        }
        return state == 1 || state == 4 || state == 6 || state == 7;
    }
    
    
    public int divide(int dividend, int divisor) {
        // Start typing your Java solution below
        // DO NOT write main() function
        int result = 0;
        long posDividend = Math.abs((long)dividend);
        long posDivisor = Math.abs((long)divisor);
        long currDivisor;
        int i = 0;
        boolean asc = true;
        while(posDividend >= posDivisor){
            currDivisor = posDivisor << i;
            if(posDividend >= currDivisor){
                posDividend -= currDivisor;
                result += 1<< i;
            }
            if(asc){
                i++;
                if(currDivisor > posDividend){
                    asc = false;
                }
            }else{
                i--;
            }
            
        }
        return ((dividend < 0)^(divisor <0)) ? (-result) : result;
        
    }
    
    public double findMedianSortedArrays(int A[], int B[]) {
        // Start typing your Java solution below
        // DO NOT write main() function
        int aLen = A.length;
        int bLen = B.length;
        if(aLen < bLen){
            return findHelper(A,B,0,aLen-1,aLen,bLen);
        }else{
            return findHelper(B,A,0,bLen-1,bLen,aLen);
        }
    }
    
    public double findHelper(int A[], int B[], int left, int right, int aLen,int bLen){
        int mid = (aLen+bLen)/2;
        if(left < right){
            return findHelper(B,A,Math.max(0,mid-aLen),Math.min(bLen-1,mid),bLen,aLen);
        }
        int i = (left+right)/2;
        int j = mid - i - 1;
        if(j>=0 && A[i]<B[j]){
           return findHelper(A,B,i+1,right,aLen,bLen);  
        }else if(j<bLen-1 && A[i] > B[j+1]){
           return findHelper(A,B,left,i-1,aLen,bLen);
        }
        if((aLen+bLen)%2 == 1) return A[i];
        if(i>0){
            return (A[i]+Math.max(B[j],A[i-1]))/2.0;
        }
        return (A[i]+B[j])/2.0;
    }
    
    public int[] plusOne(int[] digits) {
        // Start typing your Java solution below
        // DO NOT write main() function
        int originalLen = digits.length;
        int currIndex = originalLen -1;
        int currDigit = digits[currIndex];
        digits[currIndex] = (currDigit + 1)%10;
        int carry = (currDigit + 1)/10;

        for(currIndex = originalLen-2; currIndex>=0;--currIndex){
            if(carry == 0){
                break;
            }
            currDigit = digits[currIndex];
            digits[currIndex] = (currDigit + carry)%10;
            carry = (currDigit + carry)/10;
        }
        if(carry > 0){
            int[] result = new int[originalLen +1];
            result[0] = carry;
            for(int i=1;i<originalLen;++i){
                result[i] = digits[i-1];
            }
            return result;
        }else{
            return digits;
        }
    }
    
    public void solve(char[][] board) {
        // Start typing your Java solution below
        // DO NOT write main() function
        int xLen = board.length;
        if(xLen >0){
            int yLen = board[0].length;
            //top and bottom line
            for(int i=0;i<yLen;++i){
                if(board[0][i] == 'O'){
                    board = paintRecursive(board,0,i,xLen,yLen);
                }
                if(board[xLen-1][i] == 'O'){
                    board = paintRecursive(board,xLen-1,i,xLen,yLen);
                }
            }
            //left and right line
            for(int i=0;i<xLen;++i){
                if(board[i][0] == 'O'){
                    board = paintRecursive(board,i,0,xLen,yLen);
                }
                if(board[i][yLen-1] == 'O'){
                    board = paintRecursive(board,i,yLen-1,xLen,yLen);
                }
            }
            for(int i=0;i<xLen;++i){
                for(int j=0;j<yLen;++j){
                    if(board[i][j] == 'O'){
                        board[i][j] = 'X';
                    }else if(board[i][j] == '+'){
                        board[i][j] = 'O';
                    }
                }
            }
        }
        
    }
    
    public char[][] paintRecursive(char[][] board, int x, int y, int xLen, int yLen){
        if(x >= 0 && x < xLen && y>=0 && y<yLen){
                board[x][y] = '+';
                if(y > 0 && board[x][y-1] == 'O')
                board = paintRecursive(board,x,y-1,xLen,yLen);
                if(y < yLen-1 && board[x][y+1] == 'O')
                board = paintRecursive(board,x,y+1,xLen,yLen);
                if(x > 0 && board[x-1][y] == 'O')
                board = paintRecursive(board,x-1,y,xLen,yLen);
                if(x < xLen-1 && board[x+1][y] == 'O')
                board = paintRecursive(board,x+1,y,xLen,yLen);     
        }
        return board;
        
    }
    
    public boolean searchMatrix(int[][] matrix, int target) {
        // Start typing your Java solution below
        // DO NOT write main() function
        int row = matrix.length;
        if(row == 0){
            return false;
        }
        int col = matrix[0].length;
        int low = 0;
        int high = row*col-1;
        int mid;
        while(high >= low){
            mid = (high - low)/2+low;
            if(matrix[mid/col][mid%col] == target){
                return true;
            }else if(matrix[mid/col][mid%col] < target){
                low = mid+1;
            }else{
                high = mid-1;
            }
        }
        return false;
    }
    /*
    public class helperBool{
        int index;
        boolean result;
        public void helperBool(int i, boolean r) {index = i;result = r;}
    };
    */
    public boolean hasPathSum(TreeNode root, int sum) {
        // Start typing your Java solution below
        // DO NOT write main() function
        if(root == null){
            return false;
        }
        return pathSumHelper(root,0,sum);
    }
    
    public boolean pathSumHelper(TreeNode root, int currentSum, int sum){
        if(root.left == null && root.right == null){
            //it's leaf node
            return (currentSum + root.val) == sum;
        }
        boolean left = false;
        boolean right = false;
        if(root.left != null){
            left = pathSumHelper(root.left,currentSum + root.val,sum);
        }
        if(root.right!= null){
            right = pathSumHelper(root.right,currentSum + root.val,sum);
        }
        return left || right;
    }
    
    public ListNode addTwoNumbers(ListNode l1, ListNode l2) {
        // Start typing your Java solution below
        // DO NOT write main() function
        ListNode head = new ListNode(0);
        ListNode travel = head;
        ListNode iter1 = l1,iter2=l2;
        int carry = 0, curr1=0,curr2=0;
        while(iter1 != null || iter2 !=null || carry != 0){
            curr1 = iter1 == null? 0 : iter1.val;
            curr2 = iter2 == null? 0 : iter2.val;
            carry = curr1 + curr2 + carry;
            travel.next = new ListNode(carry%10);
            carry = carry/10;
            
            travel = travel.next;
            iter1 = iter1 == null? null: iter1.next;
            iter2 = iter2 == null? null: iter2.next;
        }
        
        return head.next;
    }
    
    public ArrayList<String> fullJustify(String[] words, int L) {
        // Start typing your Java solution below
        // DO NOT write main() function
        ArrayList<String> result = new ArrayList<String>();
        int listLen = words.length;
        int charLen =0,wordNum = 0;
        ArrayList<String> currentWordList = new ArrayList<String>();
        for(int i=0;i<listLen;++i){
            charLen += words[i].length();
            wordNum++;
            currentWordList.add(words[i]);
            if(charLen + wordNum -1 > L){
                currentWordList.remove(wordNum-1);
                charLen -= words[i].length();
                wordNum--;
                i--;
                
                result.add(printLine(currentWordList,L,charLen,wordNum,i == listLen-1));
                charLen = 0;
                wordNum = 0;
                currentWordList = new ArrayList<String>();
            }
        }
        if(wordNum != 0){
            result.add(printLine(currentWordList,L,charLen,wordNum,true));
        }
        return result;
    }
    
    public String printLine(ArrayList<String> words, int L, int charLen, int wordNum, boolean last){
        int spaceNum = L - charLen;
        StringBuilder sb = new StringBuilder();
        if(last){
            sb.append(words.get(0));
            for(int i=1;i<wordNum;++i){
                sb.append(' ');
                sb.append(words.get(i));
            }
            int leftSpace = L - (charLen + wordNum -1);
            for(int i=0;i<leftSpace;++i){
                sb.append(' ');
            }
        }else{
            if(wordNum == 1){
                sb.append(words.get(0));
                for(int i=0;i<spaceNum;++i){
                    sb.append(' ');
                }
            }else{
                sb.append(words.get(0));
                int spaceBetweenWords = spaceNum/(wordNum-1);
                int extraSpace = spaceNum%(wordNum-1);
                for(int i=1;i<wordNum;++i){
                    for(int j=0;j<spaceBetweenWords; ++j){
                        sb.append(' ');
                    }
                    if(i <= extraSpace){
                        sb.append(' ');
                    }
                    sb.append(words.get(i));
                }
            }
        }
        return sb.toString();
    }
    
    public int canCompleteCircuit(int[] gas, int[] cost) {
        // Note: The Solution object is instantiated only once and is reused by each test case.
        int stationNum = gas.length;
        int currGas = 0;
        int currIndex;
        for(int index = 0; index <stationNum; ++index){
            currIndex = index;
            currGas = gas[currIndex];
            currIndex++;
            while(currGas - cost[currIndex-1] >= 0){
                currGas -= cost[currIndex-1];
                if(currIndex == stationNum){
                    currIndex = 0;
                }
                if(currIndex == index){
                    return index;
                }
                currGas += gas[currIndex];
                currIndex++;
            }
        }
        return -1;
    }
    
    public ArrayList<ArrayList<Integer>> levelOrder(TreeNode root) {
        // Start typing your Java solution below
        // DO NOT write main() function
        ArrayList<ArrayList<Integer>> result = new ArrayList<ArrayList<Integer>>();
        List<TreeNode> tempList = new LinkedList<TreeNode>();
        if(root == null){
            return result;
        }
        tempList.add(root);
        ArrayList<Integer> tempResult = new ArrayList<Integer>();
        int currentCount = 1;
        int nextCount = 0;
        while(tempList.size() != 0){
            TreeNode currNode = tempList.remove(0);
            tempResult.add(currNode.val);
            currentCount--;
            if(currNode.left != null){
                tempList.add(currNode.left);
                nextCount++;
            }
            if(currNode.right != null){
                tempList.add(currNode.right);
                nextCount++;
            }
            if(currentCount == 0){
                //travel finished for current level
                result.add(tempResult);
                tempResult = new ArrayList<Integer>();
                currentCount = nextCount;
                nextCount = 0;
            }
        }
        return result;
    }
    
    
    private int solutionNum = 0;
    public int totalNQueens(int n) {
        // Start typing your Java solution below
        // DO NOT write main() function
        if(n == 1){
            return 1;
        }
        if(n == 2 || n == 3){
            return 0;
        }
        int[] previousConfig = new int[n];
        NQueenHelper(n,0,previousConfig);
        return solutionNum;
    }
    
    public void NQueenHelper(int n, int currentRow, int[] previousConfig){
        if(currentRow == n){
            //find one solution
            solutionNum++;
            return;
        }
        for(int i=0;i<n;++i){
            if(isValid(previousConfig,currentRow,i)){
                previousConfig[currentRow] = i;
                NQueenHelper(n,currentRow+1,previousConfig);
            }
        }
    }
    
    public boolean isValid(int[] previousConfig, int currentQueen, int candidatePos){
        
        for(int i=0;i<currentQueen; ++i){
            if(candidatePos == previousConfig[i]){
                return false;
            }
            if(candidatePos - currentQueen == previousConfig[i] - i || candidatePos + currentQueen == i+ previousConfig[i]){
                return false;
            }
        }
        return true;
    }
    
    public ListNode reverseBetween(ListNode head, int m, int n) {
        // Start typing your Java solution below
        // DO NOT write main() function
        ListNode dummy = new ListNode(0);
        dummy.next = head;
        ListNode travel = dummy;
        ListNode pre=new ListNode(0),swapHead=new ListNode(0),swapEnd=new ListNode(0),after=new ListNode(0);
        while(travel != null){
            m--;
            n--;
            if(m == 0){
                swapHead = travel.next;
                pre = travel;
            }
            if(n == 0){
                swapEnd = travel.next;
                after = swapEnd.next;
            }
            travel = travel.next;
        }
        reverse(swapHead,swapEnd);
        pre.next = swapEnd;
        swapHead.next = after;
        return dummy.next;
    }
    
    public void reverse(ListNode head, ListNode tail){
        ListNode nextNode = head.next;
        head.next = null;
        ListNode pre = head;
        while(pre != tail){
            
            ListNode temp = nextNode.next;
            nextNode.next = pre;
            pre = nextNode;
            nextNode = temp;
        }
    }
    
    /*
     * leetcode merge 2 sorted array.
     * 
     */
    public void merge2SortedArray(int A[], int m, int B[], int n) {
        // Start typing your Java solution below
        // DO NOT write main() function
        while(n > 0){
            if(m<=0 || A[m-1] < B[n-1]){
                A[m+n-1] = B[n-1];
                n--;
            }else{
                A[m+n-1] = A[m-1];
                m--;
            }
        }
    }
    
    /*
     * word break recursive, should work, but very inefficient
     * 
     */
   
    public boolean wordBreakRecursive(String s, Set<String> dict) {
        // Note: The Solution object is instantiated only once and is reused by each test case.
        Set<String> alreadyFound = new HashSet<String>();
        return wordBreakHelper(s,dict,alreadyFound);
        
    }
    
    public boolean wordBreakHelper(String s, Set<String> dict, Set<String> alreadyFound){
        int sLen = s.length();
        if(alreadyFound.contains(s)){
            return true;
        }
        for(int i=sLen-1;i>0;--i){
            String currSub = s.substring(0,i);
            String leftSub = s.substring(i,sLen);
            if(alreadyFound.contains(currSub)){
                if(wordBreakHelper(leftSub,dict,alreadyFound)){
                    return true;
                } 
            }
            if(dict.contains(currSub)){
                alreadyFound.add(currSub);
                if(wordBreakHelper(leftSub,dict,alreadyFound)){
                    return true;
                } 
            }
        }
        return false;
    }
    
    
    public boolean wordBreakDynamic(String s, Set<String> dict) {
        // Note: The Solution object is instantiated only once and is reused by each test case.
        int sLen = s.length();
        if(sLen == 0){
            return false;
        }
        boolean[] canBreak = new boolean[sLen];
        canBreak[0] = dict.contains(s.substring(0,1));
        for(int i=1;i<sLen;++i){
            canBreak[i] = false;
            if(dict.contains(s.substring(0,i+1))){
                canBreak[i] = true;
                continue;
            }
            for(int j=i-1;j>=0;--j){
                if(canBreak[j] && dict.contains(s.substring(j+1,i+1))){
                    canBreak[i] = true;
                }
            }
            
        }
        return canBreak[sLen-1];
    }
    
    public ArrayList<String> wordBreakII(String s, Set<String> dict) {
        // Note: The Solution object is instantiated only once and is reused by each test case.
        int sLen = s.length();
        boolean[] canBreak = new boolean[sLen];
        canBreak[0] = dict.contains(s.substring(0,1));
        for(int i=1;i<sLen;++i){
            canBreak[i] = false;
            if(dict.contains(s.substring(0,i+1))){
                canBreak[i] = true;
                continue;
            }
            for(int j=i-1;j>=0;--j){
                if(canBreak[j] && dict.contains(s.substring(j+1,i+1))){
                    canBreak[i] = true;
                }
            }
        }
        
        Set<ArrayList<String>> currResult = new HashSet<ArrayList<String>>();
        wordBreakHelper(s,dict,new ArrayList<String>(),currResult,canBreak);
        Iterator<ArrayList<String>> resultIterator = currResult.iterator();
        ArrayList<String> result = new ArrayList<String>();
        while(resultIterator.hasNext()){
            ArrayList<String> currLine = resultIterator.next();
            StringBuilder sb = new StringBuilder();
            int currentResultSize = currLine.size();
            sb.append(currLine.get(currentResultSize-1));
            for(int j=currentResultSize-2;j>=0;--j){
                sb.append(" ");
                sb.append(currLine.get(j));
            }
            result.add(sb.toString());
        }
        return result;
    }
    
    public void wordBreakHelper(String s, Set<String> dict, ArrayList<String> tempResult, Set<ArrayList<String>> result, boolean[] canBreak){
        int sLen = s.length();
        
        if(canBreak[sLen-1]){
            for(int i=sLen-1;i>=0;--i){
                if(canBreak[i]){
                    if(dict.contains(s.substring(i+1,sLen))){
                        ArrayList<String> nextResult = new ArrayList<String>(tempResult);
                        nextResult.add(s.substring(i+1,sLen));
                        wordBreakHelper(s.substring(0,i+1),dict,nextResult,result,canBreak);
                    }
                }
            }
            if(dict.contains(s)){
                tempResult.add(s);
                result.add(tempResult);
                
            }
        }
    }
    
    /*
     * some test case still can not pass, has bug in this code
     * for example: "fifgbeajcacehiicccfecbfhhgfiiecdcjjffbghdidbhbdbfbfjccgbbdcjheccfbhafehieabbdfeigbiaggchaeghaijfbjhi"
     * expected: 75, output 74
     */
    public int minCut(String s) {
        // Start typing your Java solution below
        // DO NOT write main() function
        int sLen = s.length();
        int[] minCutCount = new int[sLen];
        boolean[] hasPalindrome = new boolean[sLen];
        minCutCount[0] = 0;
        hasPalindrome[0] = false;
        for(int i=1 ;i < sLen; ++i){
            minCutCount[i] = minCutCount[i-1] + 1;
            hasPalindrome[i] = false;
            for(int j=0; j<=i-1;++j){
                if(s.charAt(j) == s.charAt(i)){
                    if(minCutIsPalindrome(s.substring(j+1,i))){
                        if(!hasPalindrome[j]){
                            minCutCount[i] = minCutCount[j];
                            hasPalindrome[i] = true;
                        }else if(minCutCount[j] < minCutCount[i-1]){
                        	minCutCount[i] = minCutCount[j] +1;
                        	hasPalindrome[i] = true;
                        }
                        break;
                    }
                }
            }
            
        }
        return minCutCount[sLen -1];
    }

    
    public boolean minCutIsPalindrome(String s){
        int startIndex = 0;
        int endIndex = s.length() - 1;
        while(startIndex < endIndex){
            if(s.charAt(startIndex) != s.charAt(endIndex)){
                return false;
            }
            startIndex++;
            endIndex--;
        }
        return true;
    }
    
    public int[] mergeSort(int[] input){
    	int inputLen = input.length;
    	if(inputLen <=1){
    		return input;
    	}
    	int leftEndIndex = inputLen/2;
    	int[] leftArray = mergeSort(Arrays.copyOfRange(input, 0, leftEndIndex));
    	int[] rightArray = mergeSort(Arrays.copyOfRange(input, leftEndIndex, inputLen));
    	return merge(leftArray,rightArray);

    }
    
    public int[] merge(int[] leftArr, int[] rightArr){
    	int leftLen = leftArr.length;
    	int rightLen = rightArr.length;
    	int resultLen = leftLen + rightLen;
    	int[] result = new int[resultLen];
    	int i=0,j=0;
    	while(i<leftLen || j<rightLen){
    		int left = i==leftLen? Integer.MAX_VALUE:leftArr[i];
    		int right = j == rightLen? Integer.MAX_VALUE:rightArr[j];
    		if(left < right){
    			result[i+j] = left;
    			i++;
    		}else if(left > right){
    			result[i+j] = right;
    			j++;
    		}else{
    			result[i+j] = left;
    			i = i==leftLen? i: i+1;
    			j = j==rightLen?j: j+1;
    		}
    	}
    	return result;
    }
    
    public String minWindow(String S, String T) {
        // Start typing your Java solution below
        // DO NOT write main() function
        Map<Character, Integer> needToFind = new HashMap<Character, Integer>();
        Map<Character, Integer> hasFound = new HashMap<Character, Integer>();
        for(int i=0;i<T.length();++i){
            //initialize needToFind map
            if(needToFind.get(T.charAt(i)) == null){
                needToFind.put(T.charAt(i),1);
            }else{
                needToFind.put(T.charAt(i),needToFind.get(T.charAt(i))+1);
            }
            if(hasFound.get(T.charAt(i)) == null){
                hasFound.put(T.charAt(i),0);
            }
        }
        int sLen = S.length();
        int tLen = T.length();
        int count = 0;
        int minLen = Integer.MAX_VALUE;
        int windowLen = 0;
        int minWindowStart =0,minWindowEnd = 0;
        for(int start = 0, end = 0; end < sLen; end++){
            if(needToFind.get(S.charAt(end)) == null){
                continue;
            }
            hasFound.put(S.charAt(end),1 + hasFound.get(S.charAt(end)));
            if(hasFound.get(S.charAt(end)) <= needToFind.get(S.charAt(end))){
                count++;
            }
            if(count == tLen){
                //all the character in T had been found in S.
                while(needToFind.get(S.charAt(start)) == null || hasFound.get(S.charAt(start)) > needToFind.get(S.charAt(start))){
                    //could advance start pointer without breaking the rules
                    if(hasFound.get(S.charAt(start)) != null && hasFound.get(S.charAt(start)) > needToFind.get(S.charAt(start))){
                        hasFound.put(S.charAt(start),hasFound.get(S.charAt(start)) -1);
                    }
                    start++;
                }
                windowLen = end - start +1;
                if(windowLen < minLen){
                    minLen = windowLen;
                    minWindowStart = start;
                    minWindowEnd = end;
                }
            }
        }
        if(count == tLen){
            return S.substring(minWindowStart,minWindowEnd+1);
        }else{
            return "";
        }
    }
    
    public int lengthOfLongestSubstringUsingSet(String s) {
        // Start typing your Java solution below
        // DO NOT write main() function
        int sLen = s.length();
        if(sLen == 0 || sLen == 1){
            return sLen;
        }
        int maxLen = 1;
        int startIndex = 0;
        int endIndex = 1;
        Set<Character> repeatSet = new HashSet<Character>();
        repeatSet.add(s.charAt(startIndex));
        while(endIndex != sLen){
            if(!repeatSet.contains(s.charAt(endIndex))){
                repeatSet.add(s.charAt(endIndex));
                endIndex++;
            }else{
                int currLen = endIndex-startIndex;
                if(currLen > maxLen){
                    maxLen = currLen;
                }
                repeatSet.clear();
                startIndex = startIndex+1;
                repeatSet.add(s.charAt(startIndex));
                endIndex = startIndex + 1;
            }
        }
        int currLen = endIndex-startIndex;
        if(currLen > maxLen){
            maxLen = currLen;
        }
        return maxLen;
    }
    
    public void reorderList(ListNode head) {
        // IMPORTANT: Please reset any member data you declared, as
        // the same Solution instance will be reused for each test case.
        
        ListNode slow = head;
        ListNode fast = head;
        ListNode reverseHead = head;
        while(fast != null && fast.next != null){
            fast = fast.next.next;
            slow = slow.next;
        }
        if(fast!= null){
            reverseHead = slow.next;
            slow.next = null;
        }else{
            reverseHead = slow;
        }
        ListNode reverseResult = reverseListIterative(reverseHead);
        ListNode travelHead = head;
        ListNode tempNext,reverseNext;
        
        //merge 2 lists one by one
        while(reverseResult != null){
            tempNext = travelHead.next;
            travelHead.next = reverseResult;
            reverseNext = reverseResult.next;
            reverseResult.next = tempNext;
            travelHead = tempNext;
            reverseResult = reverseNext;
        }
        
        
    }
    
    public ListNode reverseListIterative(ListNode curr){
        ListNode pre = null;
        ListNode temp;
        while(curr != null){
            temp = curr.next;
            curr.next = pre;
            pre = curr;
            curr = temp;
        }
        return pre;
    }
    
    public ListNode reverseListRecursive(ListNode curr, ListNode pre){
        if(curr.next == null){
            curr.next = pre;
            return curr;
        }else{
            ListNode ret = reverseListRecursive(curr.next,curr);
            curr.next = pre;
            return ret;
        }
    }
    
    public ArrayList<Integer> grayCode(int n) {
        // Start typing your Java solution below
        // DO NOT write main() function
        ArrayList<Integer> result = new ArrayList<Integer>();
        grayCodeHelper(n, result);
        return result;
    }
    
    public void grayCodeHelper(int n, ArrayList<Integer> result){
        if(n == 1){
            result.add(0);
            result.add(1);
            return;
        }
        grayCodeHelper(n-1,result);
        int preSize = result.size();
        int nextNumber = 0;
        for(int i=preSize-1;i>=0;--i){
            nextNumber = result.get(i) + (1<<n-1);
            result.add(nextNumber);
        }
        
    }
}


