# utils/auth.py
import json
import streamlit as st
import os

class Authentication:
    def __init__(self, users_file: str = "users.json"):
        self.users_file = users_file
        self.users = self.load_users()
    
    def load_users(self) -> list:
        """Load users from JSON file"""
        try:
            with open(self.users_file, 'r') as f:
                data = json.load(f)
                return data.get('users', [])
        except FileNotFoundError:
            # Create default users file
            default_users = {
                "users": [
                    {
                        "username": "user1",
                        "password": "user123",
                        "role": "user"
                    },
                    {
                        "username": "admin1",
                        "password": "admin123",
                        "role": "admin"
                    },
                    {
                        "username": "employee1",
                        "password": "employee123",
                        "role": "employee"
                    },
                    {
                        "username": "developer",
                        "password": "developer123",
                        "role": "developer"
                    }
                ]
            }
            with open(self.users_file, 'w') as f:
                json.dump(default_users, f, indent=2)
            return default_users['users']
        except json.JSONDecodeError:
            st.error("Error reading user database!")
            return []
    
    def authenticate(self, username: str, password: str):
        """Authenticate user credentials"""
        for user in self.users:
            if user['username'] == username and user['password'] == password:
                # Don't return password in user data
                user_copy = user.copy()
                if 'password' in user_copy:
                    del user_copy['password']
                return user_copy
        return None
    
    def check_session(self) -> bool:
        """Check if user is logged in"""
        return 'user' in st.session_state
    
    def get_current_user(self):
        """Get current logged in user"""
        return st.session_state.get('user')
    
    def logout(self):
        """Logout user"""
        if 'user' in st.session_state:
            del st.session_state['user']
        st.rerun()