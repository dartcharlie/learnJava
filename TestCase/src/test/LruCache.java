package test;
import java.util.*;
/*
 * 
 * need revist, when remove the first element from the doubly linked list, how do I update all the index?
 * 
 */

public class LruCache {
	private int m_capacity;
	private LinkedList<Integer> m_doublyLinkedList;
	private Map<Integer,String> m_cacheMap = new HashMap<Integer, String>();
	private Map<Integer,Object> m_listMap = new HashMap<Integer,Object>();
	private int m_currSize;
	public LruCache(int capacity){
		m_capacity = capacity;
		m_currSize = 0;
		m_doublyLinkedList =  new LinkedList<Integer>();
	}
	
	public void put(Integer i, String s){
		if(m_cacheMap.get(i) != null){
			//the record already inside the lru cache, 
			m_doublyLinkedList.remove(m_listMap.get(i));
			m_doublyLinkedList.add(i);
			m_listMap.put(i, m_doublyLinkedList.indexOf(i));
			//we could assert the doubly linked list size not changed.
			m_cacheMap.put(i, s);
		}else{
			//the record is new
			if(m_currSize < m_capacity){
				
				m_currSize++;
				//assert curr size should equal to the doubly linked list size
			}else{
				//size is at capacity
				Integer integerToRemove = m_doublyLinkedList.remove(0);
				m_listMap.remove(integerToRemove);
				m_cacheMap.remove(integerToRemove);
			}
			
			m_doublyLinkedList.add(i);
			m_listMap.put(i, m_doublyLinkedList.indexOf(i));
			m_cacheMap.put(i,s);
		}
	}
	
	public String get(Integer i){
		System.out.print("the index to remove: " + m_listMap.get(i) + " current list size: " + m_doublyLinkedList.size() + '\n');
		m_doublyLinkedList.remove(m_listMap.get(i));
		m_doublyLinkedList.add(i);
		m_listMap.put(i, m_doublyLinkedList.indexOf(i));
		return m_cacheMap.get(i);
	}
	
	public void printLinkedList(){
		for(int i=0;i<m_doublyLinkedList.size();++i){
			System.out.print(m_doublyLinkedList.get(i));
			System.out.print(',');
		}
	}
	
	
}
